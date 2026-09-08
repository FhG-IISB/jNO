"""``jno.np.cellwise(expr)`` — the per-cell L2 projection onto P0, and the B-bar it exists for.

Volumetric locking is the silent failure this closes: as ``nu -> 0.5`` (and J2 plastic flow is
isochoric, so that is plasticity's *default* regime) a standard displacement element goes rigid, and
nothing warns. B-bar replaces the volumetric strain by its cell mean and the stiffness comes back.

The oracles here are properties of the projection and of the physics, never a restatement of the
output:

  * a projection is **exact on what it projects onto** — ``cellwise`` of a constant is that constant;
  * an L2 projection **preserves the cell integral**, ``∫_K P0(f) = ∫_K f``. This is the test that
    separates the weighted mean from a plain ``jnp.mean`` over quadrature points, which does not;
  * **locking is not cured by refinement** — that is what makes it locking rather than discretization
    error. So the oracle is a refinement study: B-bar is converged from the coarsest mesh while the
    standard element is still several times too stiff many refinements later;
  * and the honest negative: on **P1 simplices the strain is already element-constant**, so the
    projection is the identity and B-bar changes nothing. Pinned so the scope cannot rot.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno.domain.geometries import Geometries

n = jno.np
sym, trace, inner, cellwise = n.symgrad, n.trace, n.inner, n.cellwise

E, DIM = 1.0, 2
L, H, BODY = 4.0, 1.0, 1.0e-3


@pytest.fixture(autouse=True)
def _x64():
    """x64: the locking contrast spans two orders of magnitude and the near-incompressible tangent is
    stiff enough that float32 cannot resolve it. The session default is x64-off (tests/conftest.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _lame(nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def _quad_beam(nx, ny):
    return jno.domain(
        constructor=Geometries.equi_distant_rect(x_range=(0, L), y_range=(0, H), nx=nx, ny=ny, cell="quad"),
        compute_mesh_connectivity=True,
    )


def _cantilever_tip(nu, nx, ny, *, bbar, domain=None):
    """Max |u_y| of a plane-strain cantilever under a body load, with or without B-bar.

    Written as the codebase writes elastic forms — a scalar contraction, no identity tensor — so the
    ONLY difference between the two branches is whether `trace(eps)` is projected.
    """
    lam, mu = _lame(nu)
    d = domain if domain is not None else _quad_beam(nx, ny)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    eps = lambda w: sym(w, [xi, yi])  # noqa: E731
    eu, ev = eps(ui), eps(vi)
    tru, trv = trace(eu), trace(ev)
    cu, cv = (cellwise(tru), cellwise(trv)) if bbar else (tru, trv)
    dev_uv = inner(eu, ev, 2) - tru * trv / DIM  # dev(eu) : dev(ev)
    mech = lam * cu * cv + 2 * mu * (dev_uv + cu * cv / DIM)  # sigma(ebar) : ebar(phi)
    fem = jno.fem([mech - BODY * vi[1], u(xl, yl) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    return float(np.abs(sol.reshape(-1, 2)[:, 1]).max())


# ----------------------------------------------------------------------------------------------
# The projection itself
# ----------------------------------------------------------------------------------------------


def test_projecting_a_constant_is_exact():
    """A projection is exact on its own range: P0 of a constant is that constant, so a form built with
    `cellwise(1)` must solve identically to one built with `1`."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    lap = ui.x * vi.x + ui.y * vi.y
    plain = np.asarray(jno.fem([lap - 1.0 * vi, u(xb, yb) - 0.0]).solve())
    projected = np.asarray(jno.fem([lap - cellwise(1.0 + 0.0 * xi) * vi, u(xb, yb) - 0.0]).solve())
    assert np.abs(plain - projected).max() < 1e-14


def test_the_projection_preserves_the_cell_integral():
    """``∫_K P0(f) = ∫_K f`` — the defining property of the L2 projection, and the one a plain
    (unweighted) mean over quadrature points does NOT have wherever the weights differ.

    Read through the load vector: Lagrange bases are a partition of unity, so ``sum_i ∫ f phi_i`` is
    ``∫_Omega f`` exactly. The mesh is quadrilateral and the field is a quadratic, so the projection is
    genuinely non-trivial per cell.
    """
    d = _quad_beam(8, 3)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 1.0 + xi * xi + 3.0 * xi * yi

    # a mass term keeps the system non-singular so `b` assembles; only `b` is read.
    b_plain = np.asarray(jno.fem([ui * vi - f * vi]).b)
    b_proj = np.asarray(jno.fem([ui * vi - cellwise(f) * vi]).b)

    assert abs(b_plain.sum() - b_proj.sum()) < 1e-12 * max(1.0, abs(b_plain.sum()))
    # ... and it is a real projection, not a no-op: the load vector itself must differ.
    assert np.abs(b_plain - b_proj).max() > 1e-6


# ----------------------------------------------------------------------------------------------
# The reason it exists: volumetric locking
# ----------------------------------------------------------------------------------------------


def test_bbar_cures_volumetric_locking_and_refinement_does_not():
    """The headline oracle. At ``nu = 0.4999`` a Q1 cantilever locks: it is an order of magnitude too
    stiff, and REFINING BARELY HELPS — which is what distinguishes locking from discretization error.
    B-bar is converged on the coarsest mesh.

    Measured here rather than asserted from theory: the B-bar answer must be stable under a 4x
    refinement in each direction, while the standard element must still be several times too stiff at
    that resolution.
    """
    nu = 0.4999
    coarse_std = _cantilever_tip(nu, 16, 4, bbar=False)
    coarse_bb = _cantilever_tip(nu, 16, 4, bbar=True)
    fine_std = _cantilever_tip(nu, 64, 16, bbar=False)
    fine_bb = _cantilever_tip(nu, 64, 16, bbar=True)

    # B-bar is converged from the coarse mesh: 4x refinement moves it by under 1%.
    assert abs(coarse_bb - fine_bb) / fine_bb < 0.01

    # The standard element is an order of magnitude too stiff on the same coarse mesh ...
    assert coarse_bb / coarse_std > 8.0
    # ... and refining it 4x does NOT rescue it — still badly short of the converged answer.
    assert fine_std < 0.5 * fine_bb


def test_bbar_changes_little_away_from_incompressibility():
    """The control that makes the test above mean what it says: at ``nu = 0.3`` there is no volumetric
    locking to cure, so the projection must be nearly inert. If B-bar moved the answer here too, the
    contrast at 0.4999 would be evidence of a bug, not of a cure."""
    std = _cantilever_tip(0.3, 16, 4, bbar=False)
    bb = _cantilever_tip(0.3, 16, 4, bbar=True)
    assert abs(bb - std) / std < 0.10


def test_bbar_is_the_identity_on_p1_simplices():
    """Scope, pinned so it cannot rot: a P1 triangle's strain is already constant over the cell, so the
    per-cell projection is the identity and B-bar changes NOTHING. This is not a defect — but it does
    mean B-bar is not the cure for P1 volumetric locking, and the docstring says so."""
    tri = jno.domain(
        constructor=Geometries.equi_distant_rect(x_range=(0, L), y_range=(0, H), nx=16, ny=4),
        compute_mesh_connectivity=True,
    )
    std = _cantilever_tip(0.4999, 0, 0, bbar=False, domain=tri)
    bb = _cantilever_tip(0.4999, 0, 0, bbar=True, domain=tri)
    # Round-off, not equality: the two assemblies take different arithmetic paths (the projection
    # averages a quantity that is already constant), and nu = 0.4999 amplifies that through the
    # conditioning -- measured 7.6e-12 relative, and it moves with the BLAS. The tolerance sits in the
    # gap between that and a REAL B-bar effect, which on a locking problem is O(1): the tests above
    # measure ratios of 1.3-2x, so there are ~9 orders between noise and signal here.
    assert abs(bb - std) <= 1e-9 * max(abs(std), 1e-30), "P1 strain is cell-constant; B-bar must be a no-op"


# ----------------------------------------------------------------------------------------------
# Fail loud
# ----------------------------------------------------------------------------------------------


def test_a_cellwise_inside_a_diff_target_raises_and_names_the_ordering():
    """`diff` is pointwise — it evaluates as `grad(sum(...))`, which is the per-point derivative only
    because the quadrature axis is a batch axis. A `cellwise` in the differentiated expression couples
    the points, so the result would silently be the cell-summed derivative."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    u, _phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    eu = sym(u.bind(x=xi, y=yi), [xi, yi])
    with pytest.raises(ValueError, match="cellwise"):
        n.diff(inner(cellwise(eu), eu, 2), eu)


def test_fbar_ordering_with_cellwise_inside_wrt_is_accepted():
    """The counterpart of the refusal above: F-bar puts the projection inside `wrt`, where substitution
    replaces it with the value slot, so it never reaches the differentiated expression. That ordering
    must be ACCEPTED — otherwise the guard would forbid the very form it recommends."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    u, _phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    eu = sym(u.bind(x=xi, y=yi), [xi, yi])
    ebar = eu * (cellwise(trace(eu)) / trace(eu))  # a cellwise living entirely inside `wrt`
    n.diff(inner(ebar, ebar, 2), ebar)  # must not raise


def test_cellwise_of_an_integral_raises():
    """An already-reduced target has no quadrature axis left to average over."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    u, _phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError, match="Integral"):
        cellwise(n.integrate(u.bind(x=xi, y=yi)))


def test_cellwise_on_a_boundary_term_raises_rather_than_averaging_the_wrong_points():
    """A surface term's kernel carries facet quadrature, not the cell's, so there is no per-cell mean to
    take. It must raise rather than average over whatever points happen to be in scope."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    vt = phi.bind(x=xt, y=yt)
    with pytest.raises(NotImplementedError, match="cellwise"):
        # the refusal fires at ASSEMBLY, which is where the missing weights are discovered
        jno.fem([ui.x * vi.x + ui.y * vi.y, cellwise(1.0 + 0.0 * xt) * vt, u(xb, yb) - 0.0]).solve()


# ----------------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------------


def test_a_cellwise_term_is_not_reported_as_spatially_local():
    """`is_local` drives operator-splitting / IMEX peeling into a node ODE. A cellwise term carries no
    spatial gradient, so the gradient-channel test alone would call it pointwise — but its value at one
    quadrature point depends on the whole cell, so peeling it would be wrong."""
    from jno.utils.solver.term_kind import classify_term

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    assert classify_term(d, ui * vi).is_local, "a plain mass term IS pointwise"
    assert not classify_term(d, cellwise(ui) * vi).is_local
    assert classify_term(d, cellwise(ui) * vi).cell_coupled
