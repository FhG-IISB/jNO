"""A periodic tie on a single-field VECTOR transient, and time-varying wall data beside a tie.

Both were refused by one message ("a periodic tie on a TRANSIENT form is supported on a scalar single
field only"), although the reduction they need already existed: the tie's node-pair weights expand
componentwise, ``kron(P_node, I_vec)``, on the steady path and on the coupled transient. The single-field
transient route now builds its reduction with the field's component count, and it no longer turns away
a time-varying Dirichlet value (its DOFs are kept out of the elimination and a destroyed row is refused
by name, as on the coupled route).

Oracles:

* a vector field whose components do not couple is **two scalar problems**: the vector march must match
  two scalar marches of the same equation to solver tolerance, and the tied seam must hold exactly;
* ``u = t (1 - y)`` solves ``u_t - Δu = 1 - y`` and lies in the P1 space and is linear in time, so
  backward Euler reproduces it to round-off, wall data ``u = t`` included.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402

PI = np.pi
inner = jno.np.inner
WALL = (1.0, -0.5)  # per-component bottom-wall values
IC = (
    lambda x, y: jnp.cos(2 * PI * x) * jnp.sin(PI * y),
    lambda x, y: jnp.sin(2 * PI * x) * jnp.sin(PI * y) + 0.3 * jnp.cos(4 * PI * x) * y * (1 - y),
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _square(t1=0.2, n=20, mesh_size=0.1):
    """Unit square, periodic in x: ``left``/``right`` include their corners, so the tie meets both walls."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, t1, n))
    d.tag("left", lambda x, y: x < 1e-6)
    d.tag("right", lambda x, y: x > 1 - 1e-6)
    d.tag("bottom", lambda x, y: y < 1e-6)
    d.tag("top", lambda x, y: y > 1 - 1e-6)
    return d


def _trajectory(fem):
    """The public solve's trajectory, on every mesh node (the reduced march is prolonged back)."""
    s = fem.solve()
    return np.asarray(s.fn() if hasattr(s, "fn") else s)


def _march(kind, comp=None):
    """``kind`` in {heat, cubic, wave}: ``u_t = ½Δu``, ``u_t = ½Δu - 0.3u³``, ``u_tt = ½Δu``.

    ``comp=None`` builds the vector field ``u ∈ R²``; ``comp=k`` the scalar problem for component k."""
    d = _square()
    vector = comp is None
    u, phi = d.fem_symbols(value_shape=(2,)) if vector else d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    x0, y0, t0 = d.variable("initial", split=True)
    ui0 = u.bind(x=x0, y=y0, t=t0)
    at = lambda tag: d.variable(tag, split=True)[:2]  # noqa: E731
    dot = (lambda a, b: inner(a, b, n_contract=1)) if vector else (lambda a, b: a * b)
    rate = ui.tt if kind == "wave" else ui.t
    pde = dot(rate, vi) + 0.5 * (dot(ui.x, vi.x) + dot(ui.y, vi.y))
    if kind == "cubic":
        pde = pde + 0.3 * dot(ui * ui * ui, vi)
    tie = u(*at("left")) - u(*at("right"))
    bottom, top = at("bottom"), at("top")  # one `d.variable` per tag: a second call samples a new tag
    if vector:
        walls = [u(*bottom)[k] - WALL[k] for k in (0, 1)] + [u(*top) - (0.0, 0.0)]
        ic = [u(x0, y0) - jno.fn(lambda x, y: jnp.stack([IC[0](x, y), IC[1](x, y)], axis=-1), [x0, y0])]
        ic += [ui0.t - (0.0, 0.0)] if kind == "wave" else []
    else:
        walls = [u(*bottom) - WALL[comp], u(*top) - 0.0]
        ic = [u(x0, y0) - jno.fn(IC[comp], [x0, y0])] + ([ui0.t - 0.0] if kind == "wave" else [])
    fem = jno.fem([pde, tie, *walls, *ic])
    Y = _trajectory(fem)
    if kind == "wave":  # the augmented state is [u; u_t]
        Y = Y[:, : Y.shape[1] // 2]
    return Y, np.asarray(fem.points)


@pytest.mark.parametrize("kind", ["heat", "cubic", "wave"])
def test_vector_transient_tie_matches_two_scalar_marches(kind):
    """A vector field periodic in x whose components do not couple: the vector march equals the two scalar
    marches of the same equation to solver tolerance, the tied seam is equal node for node, and the
    per-component wall values hold -- on the tied corners too. ``heat`` and ``cubic`` were refused before;
    ``wave`` (u_tt) already ran and is kept as the control."""
    Yv, X = _march(kind)
    U = Yv.reshape(Yv.shape[0], -1, 2)
    for k in (0, 1):
        Ys, Xs = _march(kind, comp=k)
        np.testing.assert_array_equal(X, Xs)  # same mesh, so node k of both runs is the same point
        diff = np.abs(U[..., k] - Ys).max()
        assert diff < 1e-8, f"{kind}: vector component {k} differs from its scalar march by {diff:.2e}"
    left = np.where(X[:, 0] < 1e-6)[0]
    right = np.where(X[:, 0] > 1 - 1e-6)[0]
    left, right = left[np.argsort(X[left, 1])], right[np.argsort(X[right, 1])]
    np.testing.assert_allclose(X[left, 1], X[right, 1], atol=1e-12)  # a matching seam
    assert np.abs(U[:, left] - U[:, right]).max() < 1e-14, "the tied seam is not equal node for node"
    bottom = X[:, 1] < 1e-6
    k0 = 0 if kind == "wave" else 1  # first order: the t = 0 frame is the initial condition itself
    for k in (0, 1):
        assert np.abs(U[k0:, bottom, k] - WALL[k]).max() < 1e-12, f"{kind}: wall value of component {k} not held"


@pytest.mark.parametrize(
    "vector, cubic", [(False, False), (False, True), (True, False)], ids=["scalar", "scalar-nonlinear", "vector"]
)
def test_time_varying_wall_beside_a_tie_is_exact(vector, cubic):
    """``u = t (1 - y)`` (component-scaled for a vector field) with the wall data ``u = t`` on y = 0, periodic
    in x. It is in the P1 space and linear in time, so the tied march reproduces it to round-off at every
    node, the tied corners -- which carry the time-varying value on both sides of the tie -- included. The
    nonlinear variant adds ``1e-3 (u - t(1 - y))³``, which vanishes on the solution."""
    d = _square(t1=0.3, n=12, mesh_size=0.15)
    u, phi = d.fem_symbols(value_shape=(2,)) if vector else d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    x0, y0, _t0 = d.variable("initial", split=True)
    xb, yb, tb = d.variable("bottom", split=True)
    at = lambda tag: d.variable(tag, split=True)[:2]  # noqa: E731
    scale = (1.0, -2.0) if vector else (1.0,)
    if vector:
        dot = lambda a, b: inner(a, b, n_contract=1)  # noqa: E731
        pde = dot(ui.t, vi) + dot(ui.x, vi.x) + dot(ui.y, vi.y) - (1 - yi) * (scale[0] * vi[0] + scale[1] * vi[1])
        walls = [u(xb, yb)[k] - scale[k] * tb for k in (0, 1)] + [u(*at("top")) - (0.0, 0.0)]
        ic = u(x0, y0) - (0.0, 0.0)
    else:
        pde = ui.t * vi + ui.x * vi.x + ui.y * vi.y - (1 - yi) * vi
        if cubic:
            pde = pde + 1e-3 * (ui - ti * (1 - yi)) ** 3 * vi
        walls = [u(xb, yb) - tb, u(*at("top")) - 0.0]
        ic = u(x0, y0) - 0.0
    fem = jno.fem([pde, u(*at("left")) - u(*at("right")), *walls, ic])
    Y = _trajectory(fem)
    X = np.asarray(fem.points)
    ts = np.linspace(0.0, 0.3, Y.shape[0])
    exact = ts[:, None] * (1 - X[None, :, 1])
    U = Y.reshape(Y.shape[0], -1, 2) if vector else Y[..., None]
    for k, c in enumerate(scale):
        err = np.abs(U[..., k] - c * exact).max()
        assert err < 1e-8, f"component {k}: max |u - t(1 - y)| = {err:.2e}"


def test_complex_transient_with_time_varying_wall_and_tie_is_refused_by_name():
    """The one single-field tie combination with no route: a complex transient with a time-varying
    Dirichlet value (not wired with or without a tie). It must say so, not claim ties are scalar-only."""
    d = _square(t1=0.3, n=4, mesh_size=0.3)
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    x0, y0, _t0 = d.variable("initial", split=True)
    xb, yb, tb = d.variable("bottom", split=True)
    at = lambda tag: d.variable(tag, split=True)[:2]  # noqa: E731
    with pytest.raises(NotImplementedError, match="COMPLEX transient with a time-varying Dirichlet"):
        jno.fem(
            [1j * ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - tb, u(x0, y0) - 0.0, u(*at("left")) - u(*at("right"))]
        )
