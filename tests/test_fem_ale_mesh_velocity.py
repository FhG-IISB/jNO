"""A weak form reads the mesh velocity: ``xi.d(ti)`` is w, and nodal values ride with the moving mesh.

On a moving mesh a nodal value follows its vertex, so its rate is the ALE derivative
``du/dt|_X = du/dt|_x + w . grad u``. Transport written on the moving mesh is then

    int u_t v + ((c - w) . grad u) v + nu grad u . grad v = 0,      w = (xi.d(ti), yi.d(ti)),

and the driver carries the nodal values unchanged from one configuration to the next (no L2 transfer). This
is the non-conservative ALE form, backward Euler, with the operator on the end-of-step configuration.

Oracles:

* rigid translation with w = c: the moving march IS the fixed-mesh pure-diffusion march, node for node --
  the discrete system is translation invariant and c - w vanishes exactly. A w that did not arrive, or an
  L2 transfer left on, breaks the equality by O(c dt);
* a surface term reads w as well: ``(w . n) u v`` equals ``(c . n) u v`` on that translating mesh;
* w != c: a Gaussian advected at c and diffusing, on a mesh moving at c/2, against its closed form.
"""

import jax
import numpy as np
import pytest

import jno

S2 = 0.01  # initial variance of the Gaussian
NU = 0.01
X0 = (0.35, 0.5)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _march(c, mesh_speed, *, T=0.2, n=11, size=0.1, surface=None):
    """Advection-diffusion at material velocity ``c`` on a mesh translating at ``mesh_speed``."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, T, n))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    w0, w1 = xi.d(ti), yi.d(ti)
    terms = [
        ui.t * vi + ((c[0] - w0) * ui.x + (c[1] - w1) * ui.y) * vi + NU * (ui.x * vi.x + ui.y * vi.y),
        xi.d(ti) - mesh_speed[0],  # the mesh: every vertex, both axes (a volume tag includes its boundary)
        yi.d(ti) - mesh_speed[1],
        u(ci[0], ci[1]) - jno.np.exp(-((ci[0] - X0[0]) ** 2 + (ci[1] - X0[1]) ** 2) / (2.0 * S2)),
    ]
    if surface is not None:
        xb, yb, tb, nx, ny = d.variable("boundary", normals=True, split=True)
        terms.append(surface(xb, yb, tb, nx, ny) * u.bind(x=xb, y=yb) * v.bind(x=xb, y=yb))
    return jno.fem(terms).solve()


def test_a_rigidly_translating_mesh_is_the_fixed_mesh():
    c = (1.0, 0.5)
    moving = _march(c, c)
    still = _march((0.0, 0.0), (0.0, 0.0))
    for k in (1, len(still.states) - 1):
        a, b = np.asarray(moving.states[k]), np.asarray(still.states[k])
        assert np.abs(a - b).max() <= 1e-10 * np.abs(b).max(), f"frame {k}: {np.abs(a - b).max():.3e}"
    shift = np.asarray(moving.meshes[-1][0]) - np.asarray(moving.meshes[0][0])
    assert np.allclose(shift, np.asarray(c) * 0.2, atol=1e-12), "the mesh did not translate by c*T"


def test_a_surface_term_reads_the_mesh_velocity():
    c = (1.0, 0.5)
    by_w = _march(c, c, surface=lambda xb, yb, tb, nx, ny: 0.5 * (xb.d(tb) * nx + yb.d(tb) * ny))
    by_c = _march(c, c, surface=lambda xb, yb, tb, nx, ny: 0.5 * (c[0] * nx + c[1] * ny))
    plain = _march(c, c)
    a, b = np.asarray(by_w.states[-1]), np.asarray(by_c.states[-1])
    assert np.abs(a - b).max() <= 1e-10 * np.abs(b).max(), f"{np.abs(a - b).max():.3e}"
    assert np.abs(b - np.asarray(plain.states[-1])).max() > 1e-4, "the surface term does nothing here"


def test_a_gaussian_advected_past_a_mesh_moving_at_half_its_speed():
    T = 0.3
    run = _march((1.0, 0.0), (0.5, 0.0), T=T, n=61, size=0.04)
    X = np.asarray(run.meshes[-1][0])
    u = np.asarray(run.states[-1]).reshape(-1)[: X.shape[0]]
    s2 = S2 + 2.0 * NU * T
    exact = S2 / s2 * np.exp(-((X[:, 0] - X0[0] - 1.0 * T) ** 2 + (X[:, 1] - X0[1]) ** 2) / (2.0 * s2))
    err = np.abs(u - exact).max() / exact.max()
    # A w that failed to arrive leaves the Gaussian 0.15 behind -- an O(1) error; the discretisation
    # error of this march is a few per cent.
    assert err < 0.05, f"relative max error {err:.3e}"


def _heat(d, extra, *, order=1):
    u, v = d.fem_symbols(order=order)
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    return [ui.t * vi + ui.x * vi.x + ui.y * vi.y + extra(ui, vi, xi, yi, ti), u(ci[0], ci[1]) - 1.0], (xi, yi, ti)


def _dom():
    return jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 5))


def test_the_mesh_velocity_with_no_motion_is_refused():
    terms, _ = _heat(_dom(), lambda ui, vi, xi, yi, ti: xi.d(ti) * ui.x * vi)
    with pytest.raises(ValueError, match="identically zero"):
        jno.fem(terms)


def test_the_mesh_acceleration_is_refused():
    terms, (xi, _yi, ti) = _heat(_dom(), lambda ui, vi, xi, yi, ti: xi.d(ti).d(ti) * ui * vi)
    with pytest.raises(ValueError, match="mesh acceleration"):
        jno.fem([*terms, xi.d(ti) - 0.1])


def test_the_rate_of_a_normal_is_not_a_mesh_velocity():
    d = _dom()
    xb, yb, tb, nx, _ny = d.variable("boundary", normals=True, split=True)
    u, v = d.fem_symbols()
    terms, (xi, _yi, ti) = _heat(d, lambda ui, vi, xi, yi, ti: 0.0 * ui * vi)
    with pytest.raises(ValueError, match="not a mesh coordinate"):
        jno.fem([*terms, nx.d(tb) * u.bind(x=xb, y=yb) * v.bind(x=xb, y=yb), xi.d(ti) - 0.1])


def test_a_problem_with_no_p1_field_is_refused():
    terms, (xi, _yi, ti) = _heat(_dom(), lambda ui, vi, xi, yi, ti: xi.d(ti) * ui.x * vi, order=2)
    with pytest.raises(NotImplementedError, match="P1 Lagrange"):
        jno.fem([*terms, xi.d(ti) - 0.1])
