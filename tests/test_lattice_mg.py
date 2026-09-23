"""Operator-dependent multigrid on a lattice (`jno/utils/solver/lattice_mg.py`).

The oracle is the V-cycle's own convergence factor ρ, measured as a standalone iteration
``x <- x + M(b - Ax)``: multigrid's defining property is that ρ does not grow as the grid is refined. The
V-cycle built from the grid alone (`geometric_mg`) is the comparison — it preconditions ``-Δ`` whatever
the operator is, and diverges as an iteration on every case here except Poisson.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver import lattice_mg

jax.config.update("jax_enable_x64", True)


def _problem(n, kind):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 1.0 + 0.0 * x
    jump = 1.0 + 999.0 * (x > 0.5)
    terms = {
        "poisson": [-ui.xx - ui.yy - f],
        "kappa_smooth": [-((1.0 + 10 * x**2) * ui.x).x - ((1.0 + 10 * y**2) * ui.y).y - f],
        "kappa_jump": [-(jump * ui.x).x - (jump * ui.y).y - f],
        "anisotropic": [-100.0 * ui.xx - ui.yy - f],
        "reaction": [-ui.xx - ui.yy + 500.0 * ui - f],
        "advection": [-0.01 * (ui.xx + ui.yy) + ui.x + ui.y - f],
    }[kind]
    return d, jno.fdm(terms + [u(xb, yb) - 0.0])


def _eliminated(prob):
    """The operator the structured linear path solves: the Jacobian with the Dirichlet rows and columns
    masked out, so those rows carry nothing at all."""
    residual = prob._steady_residual()
    mask = np.ones(prob._N)
    mask[prob._dirichlet_nodes()] = 0.0
    mask = jnp.asarray(mask)
    uD = jnp.zeros(prob._N)
    mv = lambda v: jax.jvp(residual, (uD,), (v * mask,))[1] * mask  # noqa: E731
    return mv, -residual(uD) * mask, mask


def _expected_levels(shape, periodic=()):
    """Every axis halves (merging an odd cell) until the level is small enough to factorise, whatever the
    grid's arithmetic: 101 nodes a side coarsen 101 -> 51 -> 26, not 101 -> 51 -> stop. A periodic axis drops
    its duplicate node first. (An isotropic operator coarsens every axis at every level.)"""
    shape = [n - 1 if a < len(periodic) and periodic[a] else n for a, n in enumerate(shape)]
    levels = 1
    while int(np.prod(shape)) > lattice_mg.DENSE_MAX:
        nxt = [
            len(lattice_mg._axis_nodes(n))
            if n >= lattice_mg.MIN_NODES
            and (not (a < len(periodic) and periodic[a]) or len(lattice_mg._axis_nodes(n)) >= 3)
            else n
            for a, n in enumerate(shape)
        ]
        if nxt == shape:
            break
        shape, levels = nxt, levels + 1
    return levels


def _rho(mv, b, M, cycles=10):
    """The convergence factor of ``x <- x + M(b - Ax)``, and the total reduction.

    Only the cycles above the round-off floor count: a V-cycle that solves exactly (a one-level hierarchy
    is a direct solve) reaches 1e-13 and then stays there, and ratios of floor values are meaningless.
    """
    x = jnp.zeros_like(b)
    norms = []
    for _ in range(cycles):
        r = b - mv(x)
        norms.append(float(jnp.linalg.norm(r)))
        x = x + M(r)
    floor = 1e-12 * norms[0]
    live = [n for n in norms if n > floor]
    if len(live) < 3:
        return 0.0, norms[-1] / norms[0]  # converged to round-off
    tail = live[-min(4, len(live)) :]
    return (tail[-1] / tail[0]) ** (1 / (len(tail) - 1)), norms[-1] / norms[0]


@pytest.mark.parametrize("kind", ["poisson", "kappa_smooth", "kappa_jump", "anisotropic", "reaction", "advection"])
def test_the_convergence_factor_does_not_grow_with_the_grid(kind):
    """ρ stays bounded and roughly constant from h to h/2 -- for a variable coefficient, a jump of 10³, a
    100:1 anisotropy, a strong reaction and an advection term, none of which the grid-only V-cycle
    preconditions (measured there: ρ 8.3, 925, 92, 22, 1.07 -- all divergent)."""
    rhos = []
    for n in (64, 128):  # both build a hierarchy: a 33² grid is below the dense coarsest level (one level)
        d, prob = _problem(n, kind)
        shape = tuple(d.mesh_connectivity["grid"]["shape"])
        mv, b, mask = _eliminated(prob)
        vcycle, levels = lattice_mg.build(mv, shape)
        if kind == "anisotropic":  # semi-coarsening: only the strong axis halves first, so it takes more levels
            assert levels >= _expected_levels(shape)
        elif kind == "advection":  # stops at or before the geometric bound: a coarse level whose smoother
            assert levels <= _expected_levels(shape)  # would amplify (cell Péclet > 2) is not built
        else:
            assert levels == _expected_levels(shape)
        rho, _ = _rho(mv, b, lambda r: vcycle(r * mask) * mask)
        rhos.append(rho)
    assert max(rhos) < 0.85, rhos
    assert rhos[1] < rhos[0] + 0.15, rhos  # no growth under refinement


def test_a_round_grid_size_coarsens_all_the_way():
    """Coarsening merges an odd cell into its neighbour, so a grid of 100 or 300 cells a side builds a full
    hierarchy. The grid-only V-cycle stopped at the first odd cell count: 1000 cells gave a 126x126
    coarsest level, and its cost depended on the grid's arithmetic rather than its size."""
    for n in (100, 300):
        d, prob = _problem(n, "poisson")
        shape = tuple(d.mesh_connectivity["grid"]["shape"])
        mv, b, mask = _eliminated(prob)
        vcycle, levels = lattice_mg.build(mv, shape)
        assert levels == _expected_levels(shape), (n, levels)
        rho, _ = _rho(mv, b, lambda r: vcycle(r * mask) * mask)
        assert rho < 0.6, (n, rho)


def test_the_v_cycle_is_symmetric_when_the_operator_is():
    """A Galerkin coarse operator with Pᵀ restriction and matching pre/post smoothing is symmetric, which is
    what makes it a valid preconditioner for conjugate gradients."""
    d, prob = _problem(24, "kappa_smooth")
    shape = tuple(d.mesh_connectivity["grid"]["shape"])
    mv, _, mask = _eliminated(prob)
    vcycle, _ = lattice_mg.build(mv, shape)
    rng = np.random.default_rng(0)
    v, w = (jnp.asarray(rng.standard_normal(prob._N)) * mask for _ in range(2))
    a, b = float(w @ vcycle(v)), float(v @ vcycle(w))
    assert abs(a - b) <= 1e-10 * max(abs(a), abs(b))


def test_it_preconditions_the_solve_to_the_tolerance():
    """The measure that matters: the preconditioned Krylov solve reaches the tolerance, and its answer is
    the direct solve's."""
    from jno.fdm import _gmres_incremental

    d, prob = _problem(32, "kappa_jump")
    shape = tuple(d.mesh_connectivity["grid"]["shape"])
    mv, b, mask = _eliminated(prob)
    vcycle, _ = lattice_mg.build(mv, shape)
    x, rn = _gmres_incremental(mv, b, lambda r: vcycle(r * mask) * mask, 1e-12)
    assert float(rn) / float(jnp.linalg.norm(b)) < 1e-11
    dense = jax.jacfwd(mv)(jnp.zeros(prob._N))
    live = np.abs(np.asarray(dense)).sum(axis=1) > 0
    exact = np.zeros(prob._N)
    sub = np.asarray(dense)[np.ix_(live, live)]
    exact[live] = np.linalg.solve(sub, np.asarray(b)[live])
    np.testing.assert_allclose(np.asarray(x), exact, atol=1e-8 * np.abs(exact).max())


def test_a_periodic_axis_keeps_its_wrap_on_every_level():
    """A periodic grid problem: the hierarchy wraps the same axis all the way down, and ρ stays bounded.
    The grid-only V-cycle was skipped entirely on a periodic grid (plain GMRES, no preconditioner)."""
    import jno.jnp_ops as jnn

    n = 32
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 5 * np.pi**2 * jnn.sin(2 * np.pi * x) * jnn.sin(np.pi * y)
    prob = jno.fdm([-ui.xx - ui.yy - f, u(xl, yl) - u(xr, yr), u(xb, yb) - 0.0, u(xt, yt) - 0.0])
    shape = tuple(d.mesh_connectivity["grid"]["shape"])
    mv, b, mask = _eliminated(prob)
    vcycle, levels = lattice_mg.build(mv, shape, periodic=(True, False))
    assert levels == _expected_levels(shape, (True, False))
    rho, _ = _rho(mv, b, lambda r: vcycle(r * mask) * mask)
    assert rho < 0.9, rho
