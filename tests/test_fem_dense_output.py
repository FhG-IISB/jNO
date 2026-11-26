"""Dense output: a transient sampled at ``save_ts`` between its own steps keeps the scheme's order.

The adaptive march takes its own steps and samples the requested times afterwards. A linear blend of the
two bracketing states is only second order, so a third-order scheme taking large steps returned output
far less accurate than its steps (Hairer, Norsett & Wanner I, Sec. II.6). The oracle is a fine fixed-step
SDIRK3 march whose grid CONTAINS every save time (no interpolation in the reference).

Run with x64 (the FEM assembly is float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
T1 = 0.05


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nsteps, mesh_size=0.25):
    """u_t = lap u on the unit square, u0 = sin(pi x) sin(pi y) + sin(2 pi x) sin(3 pi y) (two decay rates,
    so the trajectory is not a single exponential), zero Dirichlet."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, T1, nsteps))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    s = jno.np.sin
    u0 = s(PI * ci[0]) * s(PI * ci[1]) + s(2 * PI * ci[0]) * s(3 * PI * ci[1])
    return jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


N_SAVE = 81  # save times every T1/80; the reference grid (1601 points) contains all of them
SAVE = jnp.linspace(0.0, T1, N_SAVE)


@pytest.fixture(scope="module")
def reference():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        fem = _heat(nsteps=1601)
        grid = jnp.linspace(0.0, T1, 1601)  # the domain's own grid: sampled without interpolation
        full = np.asarray(jno.solve.sdirk(3).integrate(fem.operator, {}, grid, linear_solve=None, nonlinear_solve=None))
        assert full.shape[0] == 1601
        return full[::20]  # rows at k * T1/80, exactly on the grid
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("scheme", [jno.solve.sdirk(3), jno.solve.rosenbrock("ros34pw2")], ids=["sdirk3", "ros34pw2"])
def test_adaptive_output_between_steps_is_as_accurate_as_at_the_steps(reference, scheme):
    """A loose tolerance gives a few large steps; nearly every save time falls between two of them. The
    worst error over all save times may exceed the error at the final time (a step node) by a small
    factor, not by the orders of magnitude a linear blend costs."""
    fem = _heat(nsteps=6)
    out = np.asarray(
        scheme.adaptive(rtol=1e-6, atol=1e-9).integrate(fem.operator, {}, SAVE, linear_solve=None, nonlinear_solve=None)
    )
    assert not np.any(np.isnan(out))
    at_node = _rel(out[-1], reference[-1])
    worst = max(_rel(out[i], reference[i]) for i in range(1, N_SAVE))
    assert worst < 3 * max(at_node, 1e-9), f"between steps {worst:.2e} vs at the last step {at_node:.2e}"


def test_dense_output_keeps_the_gradient_finite(reference):
    """The interpolant is a polynomial in the stored states, so the parameter gradient flows through it
    (and a rejected attempt's duplicate time must not divide by zero)."""
    fem = _heat(nsteps=6)
    scheme = jno.solve.sdirk(3).adaptive(rtol=1e-5, atol=1e-8)

    def loss(scale):
        out = scheme.integrate(fem.operator, {}, SAVE, linear_solve=None, nonlinear_solve=None)
        return jnp.sum((scale * out) ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g)) and float(g) > 0


MID = jnp.asarray((np.arange(40) + 0.5) * T1 / 40)  # midpoints of a 40-step grid; on the reference grid


@pytest.mark.parametrize(
    "scheme,ids", [(jno.solve.sdirk(3), "sdirk3"), (jno.solve.bdf2(), "bdf2"), (jno.solve.theta(0.5), "cn")]
)
def test_fixed_grid_midpoints_keep_the_scheme_order(reference, scheme, ids):
    """A fixed 40-step march sampled halfway between its steps: the degree-p interpolant keeps the error
    near the error at the steps. For the third-order ``sdirk(3)`` the linear blend gave 29 times the step
    error here; for the second-order ``bdf2`` / ``theta(0.5)`` the blend is already of their order, and
    these cases guard that the quadratic does not make them worse."""
    fem = _heat(nsteps=41)
    nodes = np.asarray(
        scheme.integrate(fem.operator, {}, jnp.linspace(0.0, T1, 41), linear_solve=None, nonlinear_solve=None)
    )
    mid = np.asarray(scheme.integrate(fem.operator, {}, MID, linear_solve=None, nonlinear_solve=None))
    ref_nodes, ref_mid = reference[::2], reference[1::2]  # reference rows every T1/80: even = nodes, odd = midpoints
    at_nodes = max(_rel(nodes[i], ref_nodes[i]) for i in range(1, 41))
    between = max(_rel(mid[i], ref_mid[i]) for i in range(40))
    assert between < 3 * at_nodes, f"{ids}: between steps {between:.2e} vs at the steps {at_nodes:.2e}"


@pytest.mark.parametrize("scheme", [jno.solve.sdirk(3), jno.solve.bdf2()], ids=["sdirk3", "bdf2"])
def test_eager_and_traced_marches_sample_alike(scheme):
    """The eager march keeps only the states its frames read (``_needed_steps``) and samples on the host;
    the traced march samples the whole trajectory on the device. Off the grid both must use the same
    window and weights."""
    fem = _heat(nsteps=41)
    eager = np.asarray(scheme.integrate(fem.operator, {}, MID, linear_solve=None, nonlinear_solve=None))
    traced = np.asarray(
        jax.jit(lambda s: scheme.integrate(fem.operator, {}, s, linear_solve=None, nonlinear_solve=None))(MID)
    )
    assert np.max(np.abs(eager - traced)) < 1e-12 * np.max(np.abs(traced))


def test_needed_steps_mark_the_interpolation_window():
    from jno.utils.solver.backend_blocks import _needed_steps

    grid = np.linspace(0.0, 1.0, 11)
    assert list(_needed_steps(grid, np.asarray([0.55]))) == [5, 6]  # degree 1: the bracketing pair
    assert list(_needed_steps(grid, np.asarray([0.55]), 3)) == [4, 5, 6, 7]  # cubic: centred window
    assert list(_needed_steps(grid, np.asarray([0.95]), 3)) == [7, 8, 9, 10]  # shifted inside at the end
    assert list(_needed_steps(grid, np.asarray([0.5]), 3)) == [5]  # on the grid: the point itself


def test_a_chunked_eager_march_keeps_the_order_at_chunk_boundaries(reference, monkeypatch):
    """A march whose frames outgrow the device runs in chunks; a frame next to a chunk boundary reads the
    states kept from the chunk before (left) and a one-sided window (right). Forced here with a budget of
    3 frames: the midpoints must still be about as accurate as the steps."""
    import jno.utils.solver.backend_blocks as bb

    monkeypatch.setattr(bb, "_offload_chunk", lambda n_steps, n_dofs, dtype: 3)
    scheme = jno.solve.sdirk(3)
    fem = _heat(nsteps=41)
    mid = np.asarray(scheme.integrate(fem.operator, {}, MID, linear_solve=None, nonlinear_solve=None))
    nodes = np.asarray(
        scheme.integrate(fem.operator, {}, jnp.linspace(0.0, T1, 41), linear_solve=None, nonlinear_solve=None)
    )
    at_nodes = max(_rel(nodes[i], reference[::2][i]) for i in range(1, 41))
    between = max(_rel(mid[i], reference[1::2][i]) for i in range(40))
    assert between < 3 * at_nodes, f"chunked: between steps {between:.2e} vs at the steps {at_nodes:.2e}"
