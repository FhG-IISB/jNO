"""``fem.eval(expr, u)`` on a **test-free** expression — the domain integral ``∫ F(u, ∇u, ρ, x) dΩ``.

The sibling of the reaction readout in ``test_fem_eval_readout.py``. A term carrying a test function
assembles to one value per DOF; an expression *without* one is an energy/stress density, and its
natural reduction is the integral over the region it lives on. That is the objective a design problem
minimises (compliance, volume fraction, a stress p-norm), so it has to be differentiable in the
solution, in the design parameters, and in the mesh coordinates.

Oracles are independent of the assembly being tested:

* closed-form integrals of polynomials over a rectangle;
* the **partition of unity**: ``Σ_a φ_a ≡ 1`` for a Lagrange basis, so ``sum(fem.eval(F*φ, u))``
  is the same integral assembled through the *weak-term* path — a route that already existed;
* the **bilinear form at its own solution**: ``∫ σ(u):ε(u) dΩ == u · fem.eval(a(u,φ), u)``, which
  holds exactly only if the functional inherits the quadrature the assembly used;
* the divergence theorem, for the boundary measure;
* central finite differences, for the mesh derivative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("basix", reason="native Lagrange assembler needs basix")

E0, NU = 200.0, 0.3
LAM = E0 * NU / ((1 + NU) * (1 - 2 * NU))
MU = E0 / (2 * (1 + NU))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.25, w=2.0, h=1.0):
    """A solved Poisson problem on ``[0,w]x[0,h]``, plus the pieces a functional is written from."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, w, h, size=size).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > w - 1e-9) | (y < 1e-9) | (y > h - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(*cw) - 0.0])
    return d, fem, u, phi, X, fem.solve()


# ---------------------------------------------------------------- the volume measure


def test_the_integral_of_one_is_the_domain_measure():
    """The cheapest oracle there is: ∫ 1 dΩ over a 2x1 rectangle is its area."""
    _d, fem, _u, _phi, X, sol = _poisson()
    area = float(np.asarray(fem.eval(0.0 * X[0] + 1.0, sol)).reshape(-1)[0])
    assert abs(area - 2.0) < 1e-12, f"∫1 dΩ = {area:.15f}, expected 2.0"


@pytest.mark.parametrize(
    "make, exact",
    [
        (lambda X: X[0], 2.0),  # ∫x over [0,2]x[0,1] = w^2 h / 2
        (lambda X: X[1], 1.0),  # ∫y = w h^2 / 2
        (lambda X: X[0] * X[1], 1.0),  # ∫xy = w^2 h^2 / 4
    ],
    ids=["x", "y", "xy"],
)
def test_polynomial_integrals_match_their_closed_forms(make, exact):
    """Affine geometry and a degree->=2 rule integrate these exactly, so the tolerance is round-off."""
    _d, fem, _u, _phi, X, sol = _poisson()
    got = float(np.asarray(fem.eval(make(X), sol)).reshape(-1)[0])
    assert abs(got - exact) < 1e-12, f"got {got:.15f}, exact {exact}"


def test_it_agrees_with_the_partition_of_unity_assembly():
    """Σ_a φ_a ≡ 1, so summing the weak term ``F*φ`` over every DOF is the same integral by a route
    that already worked. Uses a NON-polynomial F, where the two must agree on the same quadrature
    rather than both happening to be exact."""
    _d, fem, _u, phi, X, sol = _poisson()
    F = jno.np.exp(X[0] * X[1]) + jno.np.sin(3.0 * X[0])
    direct = float(np.asarray(fem.eval(F, sol)).reshape(-1)[0])
    via_weak = float(jnp.sum(jnp.asarray(fem.eval(F * phi, sol))))
    assert abs(direct - via_weak) < 1e-10, f"functional {direct:.12f} vs weak-term sum {via_weak:.12f}"


def test_the_integral_of_the_solution_is_differentiable_in_it():
    """A functional of the solved field carries a gradient back to the DOFs — the property that makes
    it usable as an objective at all."""
    _d, fem, u, _phi, _X, sol = _poisson()
    g = jax.grad(lambda w: jnp.asarray(fem.eval(u * u, w)).reshape(-1)[0])(jnp.asarray(sol))
    assert np.isfinite(np.asarray(g)).all()
    assert float(jnp.abs(g).max()) > 0.0


# ---------------------------------------------------------------- elasticity: the compliance identity


def _elastic(size=0.5, w=6.0, h=3.0):
    """A plane-strain cantilever whose stiffness carries a P0 density.

    Returns the domain, the FEM, the density symbol, the weak term, and the strain-energy integrand
    ``σ(u):ε(u)`` written with the SAME bilinear form -- so the two sides of the compliance identity
    are demonstrably the same expression, not two hand-typed ones that might differ."""
    inner, sym, tr = jno.np.inner, jno.np.symgrad, jno.np.trace
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731
    d = jno.Shape.rect(0.0, 0.0, w, h, size=size).domain()
    d.tag("root", lambda x, y: x < 1e-9)
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("root", split=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    rho = jno.np.parameter(s, name="rho")
    rho.dtype(jnp.float64)
    eu, ep = sym(u, [xi, yi]), sym(phi, [xi, yi])
    a = lambda pp, qq: LAM * tr(pp) * tr(qq) + 2.0 * MU * ddot(pp, qq)  # noqa: E731  -- sigma(pp) : qq
    mech = rho * a(eu, ep)
    body = -1.0 * inner(jnp.array([0.0, -1.0]), phi.bind(x=xi, y=yi), 1)
    fem = jno.fem([mech, u(xl, yl) - (0.0, 0.0), body])
    return d, fem, rho, mech, rho * a(eu, eu)


def _solve_at(fem, args):
    """Concrete DOFs for a runtime-parametric system, by sparse-direct factorisation of the assembled
    operator — the same route the topology-optimisation tutorial's reanalysis uses."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    mat, rhs = fem.operator.evaluate(args)
    f = np.asarray(jnp.asarray(rhs).reshape(-1), dtype=np.float64)
    idx = np.asarray(mat.indices)
    k = sp.csr_matrix((np.asarray(mat.data, dtype=np.float64), (idx[:, 0], idx[:, 1])), shape=(f.size, f.size))
    return spla.spsolve(k.tocsc(), f)


def _cell_volumes(d):
    cells, pts = np.asarray(d._cells_p1()), np.asarray(d.mesh.points)[:, :2]
    v = pts[cells]
    return 0.5 * np.abs(np.linalg.det(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], -1)))


def test_the_strain_energy_functional_equals_the_bilinear_form_at_the_solution():
    """C = ∫ σ(u):ε(u) dΩ = a(u,u) = u·(Ku). The right-hand side is the existing weak-term readout, so
    this pins that the functional inherits the SAME quadrature the assembly used — on a different rule
    the two would differ by a quadrature error and nothing would say so."""
    d, fem, _rho, mech, energy = _elastic()
    args = {"rho": jnp.full(np.asarray(d._cells_p1()).shape[0], 0.5)}
    sol = _solve_at(fem, args)
    lhs = float(np.asarray(fem.eval(energy, sol, args=args)))
    rhs = float(np.asarray(fem.eval(mech, sol, args=args)).reshape(-1) @ sol)
    assert abs(lhs - rhs) / abs(rhs) < 1e-11, f"∫σ:ε = {lhs:.12f} vs u·Ku = {rhs:.12f}"


def test_a_one_hot_p0_density_integrates_to_that_element_volume():
    """A per-element design value must land on its own element and nowhere else — the P0 contract,
    read through the integral."""
    d, fem, rho, _mech, _energy = _elastic()
    vols = _cell_volumes(d)
    sol = _solve_at(fem, {"rho": jnp.full(vols.size, 0.5)})
    k = 7
    got = float(np.asarray(fem.eval(rho, sol, args={"rho": jnp.zeros(vols.size).at[k].set(1.0)})))
    assert abs(got - vols[k]) < 1e-12, f"∫ρ dΩ = {got:.15f}, element {k} volume {vols[k]:.15f}"


def test_a_uniform_density_integrates_to_a_fraction_of_the_domain_measure():
    d, fem, rho, _mech, _energy = _elastic()
    n = np.asarray(d._cells_p1()).shape[0]
    sol = _solve_at(fem, {"rho": jnp.ones(n)})
    got = float(np.asarray(fem.eval(rho, sol, args={"rho": jnp.full(n, 0.25)})))
    assert abs(got - 0.25 * 18.0) < 1e-11, f"∫0.25 dΩ = {got:.12f}, expected {0.25 * 18.0}"


def test_the_functional_is_differentiable_in_the_design():
    """d/dρ of ∫ρ dΩ is the element-volume vector — an exact gradient oracle, not a finite difference."""
    d, fem, rho, _mech, _energy = _elastic()
    vols = _cell_volumes(d)
    sol = _solve_at(fem, {"rho": jnp.full(vols.size, 0.5)})
    g = jax.grad(lambda rv: jnp.asarray(fem.eval(rho, sol, args={"rho": rv})).reshape(()))(jnp.full(vols.size, 0.5))
    assert np.abs(np.asarray(g) - vols).max() < 1e-12


# ---------------------------------------------------------------- the boundary measure


def test_a_boundary_functional_measures_the_region_it_lives_on():
    """∫ 1 ds over the tagged right edge of a 2x1 rectangle is that edge's length."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.25).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 2 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    d.tag("east", lambda x, y: x > 2 - 1e-9)
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    ce = d.variable("east", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(*cw) - 0.0])
    sol = fem.solve()
    length = float(np.asarray(fem.eval(0.0 * ce[0] + 1.0, sol)).reshape(-1)[0])
    assert abs(length - 1.0) < 1e-12, f"∫1 ds = {length:.15f}, expected 1.0 (the east edge)"


def test_a_boundary_flux_functional_matches_the_divergence_theorem():
    """∮ F·n ds = ∫ div F dΩ. With F = (x, y), div F = 2, so the closed boundary integral is 2|Ω|."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.2).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 2 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co = d.variable("interior", split=True)
    cw = d.variable("walls", normals=True, split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(*cw[:2]) - 0.0])
    sol = fem.solve()
    xb, yb, nx, ny = cw[0], cw[1], cw[-2], cw[-1]
    flux = float(np.asarray(fem.eval(xb * nx + yb * ny, sol)).reshape(-1)[0])
    assert abs(flux - 2.0 * 2.0) < 1e-10, f"∮F·n ds = {flux:.12f}, expected {2.0 * 2.0}"


# ---------------------------------------------------------------- the mesh derivative


def test_the_functional_is_differentiable_in_the_mesh_coordinates():
    """The property a deformable-mesh design problem needs: ∂/∂X must flow through |det J|.
    Oracle is a central finite difference on the same functional."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 2 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    xm, ym, _ = d.variable("mv", where=lambda x, y: (x > 1e-9) & (x < 2 - 1e-9) & (y > 1e-9) & (y < 1 - 1e-9), split=True)
    xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y")
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(*cw) - 0.0])

    pts0 = np.asarray(d.mesh.points)[:, :2]
    ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
    coord0 = {sp["name"]: jnp.asarray(pts0[ids, int(sp["axis"])]) for sp in d._trainable_coords}
    sol = _solve_at(fem, coord0)

    # ∫ x^2 dΩ: a functional with no field in it, so the ONLY route for a gradient is the geometry.
    f = lambda a: jnp.asarray(fem.eval(X[0] * X[0], sol, args=a)).reshape(-1)[0]  # noqa: E731
    g = jax.grad(f)(coord0)["mesh_x"]

    i, eps = 0, 1e-6
    plus = dict(coord0, mesh_x=coord0["mesh_x"].at[i].add(eps))
    minus = dict(coord0, mesh_x=coord0["mesh_x"].at[i].add(-eps))
    fd = (float(f(plus)) - float(f(minus))) / (2 * eps)
    assert abs(float(g[i]) - fd) < 1e-6, f"AD {float(g[i]):.9f} vs FD {fd:.9f}"


# ---------------------------------------------------------------- scope limits, stated loudly


def test_a_functional_spanning_two_regions_is_refused():
    """An integral has one measure. A term touching both the volume and a boundary tag names neither,
    so it must be refused rather than silently integrated over one of them."""
    d, fem, _u, _phi, X, sol = _poisson()
    cw = d.variable("walls", split=True)
    with pytest.raises(ValueError, match="single region"):
        fem.eval(X[0] * cw[0], sol)


def test_a_mixed_list_of_weak_terms_and_integrands_is_refused():
    """They reduce to different shapes — one value per DOF versus one scalar — so a mixed list has no
    single return type."""
    _d, fem, _u, phi, X, sol = _poisson()
    with pytest.raises(ValueError, match="same kind"):
        fem.eval([X[0] * phi, X[0]], sol)
