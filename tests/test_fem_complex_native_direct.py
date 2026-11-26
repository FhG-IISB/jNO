"""A direct backend that factors complex matrices solves a complex form's own ``n x n`` operator.

``jno.solve.lu(backend="pardiso")`` hands MKL PARDISO ``A_r + i A_i`` (complex mtypes 13 / 6 / -4) rather
than the real-equivalent ``2n`` block ``[[A_r, -A_i], [A_i, A_r]]``: the factor needs about half the
memory, and the periodic reduction and the host copies run on ``n``-sized legs. The answer must be the
one the real-equivalent block gives (``backend="host"``, SuperLU on the 2n block) -- with periodic and
Bloch ties, with a prescribed value the tie reduction has to put back, and through a gradient.
"""

import numpy as np
import pytest

pytest.importorskip("pypardiso", reason="MKL PARDISO (pypardiso) required")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver import linear as _lin  # noqa: E402

K = 2 * np.pi
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve(fem, backend):
    return np.asarray(fem.solve(linear=jno.solve.lu(backend=backend))).reshape(-1)


def _strip(P=1.3, L=0.6, h=0.1, phase=None):
    """Helmholtz in a periodic strip (Bloch when ``phase`` is given): ``u = cos(k_x x)`` at y = 0, the
    first-order absorbing face on top."""
    d = jno.shape.rect(0.0, 0.0, P, L, size=h).domain()
    e = 1e-6
    d.tag("bottom", lambda x, y: y < e)
    d.tag("top", lambda x, y: y > L - e)
    d.tag("left", lambda x, y: x < e)
    d.tag("right", lambda x, y: x > P - e)
    u, v = d.fem_symbols()
    X = {n: d.variable(n, split=True) for n in ("interior", "bottom", "top", "left", "right")}
    xi, yi, _ = X["interior"]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xt, yt, _ = X["top"]
    xb, yb, _ = X["bottom"]
    g = jno.np.cos(2 * np.pi / P * xb)  # a real essential value (a complex one is refused by design)
    right = u(*X["right"][:2]) if phase is None else phase * u(*X["right"][:2])
    return jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - K**2 * ui * vi,
            -1j * K * u.bind(x=xt, y=yt) * v.bind(x=xt, y=yt),
            u(xb, yb) - g,
            u(*X["left"][:2]) - right,
        ]
    )


def test_pardiso_factors_the_complex_operator_and_matches_the_real_equivalent_block():
    _lin._PARDISO_CACHE.clear()
    fem = _strip()
    ref, got = _solve(fem, "host"), _solve(fem, "pardiso")
    assert np.abs(ref).max() > 0.5
    np.testing.assert_allclose(got, ref, atol=1e-11)
    entries = list(_lin._PARDISO_CACHE.values())
    assert entries and all(e["mtype"] in (13, 6, -4) for e in entries), [e["mtype"] for e in entries]
    assert all(np.iscomplexobj(e["A"].data) for e in entries)


def test_bloch_tie_on_the_complex_operator():
    """A Bloch tie's complex prolongation reduces the complex operator Hermitian-wise (P^H A P)."""
    P = 1.3
    fem = _strip(P=P, phase=np.exp(-1j * 0.4 * K * P))
    ref, got = _solve(fem, "host"), _solve(fem, "pardiso")
    assert np.abs(ref).max() > 0.5
    np.testing.assert_allclose(got, ref, atol=1e-11)


def test_a_value_on_a_tied_corner_is_put_back_on_the_complex_operator():
    """A value written on the corner every tie of a doubly periodic box eliminates: the reduction sums
    that row into its partners, and the complex route must restore it like the real-equivalent one."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=12).domain()
    e = 1e-9
    for nm, pr in {
        "left": lambda x, y: x < e,
        "right": lambda x, y: x > 1 - e,
        "bottom": lambda x, y: y < e,
        "top": lambda x, y: y > 1 - e,
    }.items():
        d.tag(nm, pr)
    at = lambda t: d.variable(t, split=True)[:2]  # noqa: E731
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = (1.0 + 0.5j) * jno.np.cos(2 * np.pi * xi) * jno.np.cos(2 * np.pi * yi)
    d.point_region("corner", (0.0, 0.0))
    fem = jno.fem(
        [
            (1.0 + 0.3j) * (ui.x * vi.x + ui.y * vi.y) + 0.2j * ui * vi - f * vi,
            u(*at("left")) - u(*at("right")),
            u(*at("bottom")) - u(*at("top")),
            u(*at("corner")) - 0.0,
        ]
    )
    assert fem._periodic.get("dirichlet_reduced"), "this case must exercise the reduced-row restoration"
    ref, got = _solve(fem, "host"), _solve(fem, "pardiso")
    assert np.abs(ref.imag).max() > 1e-3
    np.testing.assert_allclose(got, ref, atol=1e-11)
    pts = np.asarray(fem.points)[:, :2]
    on = lambda a: (np.abs(a) < 1e-9) | (np.abs(a - 1.0) < 1e-9)  # noqa: E731
    np.testing.assert_allclose(got[on(pts[:, 0]) & on(pts[:, 1])], 0.0, atol=1e-12)


def test_gradient_flows_through_the_complex_factorisation():
    """A real parameter in a complex form, recovered from full-field data with the solve on the complex
    operator: the adjoint is the transpose solve of the same complex factorisation."""
    from shapely.geometry import box

    true = 0.8

    def form(kappa):
        dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
        u, v = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        g = jno.np.cos(np.pi * xi) * jno.np.cos(np.pi * yi)
        f = (2 * np.pi**2 * true + 1.0 + 0.3j) * (1.0 + 0.5j) * g
        return jno.fem([kappa * (ui.x * vi.x + ui.y * vi.y) + (1.0 + 0.3j) * (u * vi) - f * vi])

    u_obs = _solve(form(true), "host")
    kappa = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="kappa")
    kappa.initialize(jax.nn.initializers.constant(1.5))
    kappa.dtype(jnp.float64)
    kappa.optimizer(optax.adam(5e-2))
    node = form(kappa).solve(linear=jno.solve.lu(backend="pardiso"))
    crux = jno.core([(node - u_obs).mae], domain=_DUMMY)
    crux.solve(250)
    assert abs(float(np.asarray(crux.eval([kappa])).reshape(-1)[0]) - true) < 0.05
