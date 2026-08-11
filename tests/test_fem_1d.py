"""1D ("segment" / ``LINE2``) FEM coverage through ``jno.fem``.

1D is assembled by a small dedicated ``LINE2``
assembler (``jno/utils/solver/fem_1d.py``) that reuses the same integrand
evaluator as the 2D/3D path. These tests mirror ``test_fem_3d.py`` on a line
domain (``jno.domain.line`` -> pygmsh): steady (linear + nonlinear, all BCs) and
transient (1D+time). Same matrices-only contract — no solve.

1D linear FEM is *nodally exact* for ``-u'' = f`` (the discrete Green's function
reproduces nodal values), so the linear Dirichlet/Neumann/Robin cases recover to
machine precision; the nonlinear and transient cases use mesh-appropriate tols.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("pygmsh", reason="pygmsh required for line meshing")

import jax  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64, so these tests opt into x64 per-test. The session default is
    x64-off (see tests/conftest.py); save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _solve(fem):
    return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))


def _line(mesh_size=0.05, **kwargs):
    return jno.domain(constructor=jno.domain.line(mesh_size=mesh_size), **kwargs)


def _x(d):
    return np.asarray(d.mesh.points)[:, 0]


# ==========================================================================
# structure
# ==========================================================================
def test_assembly_is_sparse_and_scales_past_the_dense_ceiling():
    """1D assembles into a **BCOO**, scattered from per-element blocks.

    It used to recover the global operator as ``jacfwd(R)(0)`` over the whole residual, which made 1D —
    the *cheapest* dimension — carry the library's lowest DOF ceiling: the ``(n_elem, n_dof, ...)``
    intermediate exhausted GPU memory at ~10k nodes while 2D/3D scattered sparsely all along.

    Pins the structure (BCOO, and ``nnz`` growing **linearly**, which a dense or dense-derived operator
    could not do), and solves at a node count that used to be unreachable."""
    d = _line(0.001)  # ~1000 nodes: comfortably past nothing, but pins the nnz law
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x - 1.0 * vi, u(xb) - 0.0])
    A, _b = fem.operator
    n = A.shape[0]
    assert hasattr(A, "indices"), "1D must assemble sparsely (BCOO), not into a dense array"
    # a LINE2 element contributes a 2x2 block, plus one entry per Dirichlet row: nnz ~ 4*(n-1) + 2,
    # i.e. LINEAR in n. A dense N^2 operator at n=1001 would be 8 MB of mostly zeros.
    assert int(A.nse) < 8 * n, f"nnz={int(A.nse)} is not linear in n={n} — the scatter is not element-local"

    x = np.asarray(d.mesh.points)[:, 0]
    sol = np.asarray(fem.solve())
    exact = x * (1.0 - x) / 2.0  # -u'' = 1, u(0)=u(1)=0
    assert np.max(np.abs(sol - exact)) < 1e-11, "the sparse assembly must stay nodally exact"


def test_line_domain_is_1d():
    d = _line(0.2)
    assert d.dimension == 1
    assert "line" in d.mesh.cells_dict
    assert {"left", "right", "boundary"} <= set(getattr(d, "_boundary_regions", {}))


# ==========================================================================
# vector unknowns (value_shape=(n,)) — systems on a line
# ==========================================================================
def _vector_system(ms=0.02, order=1):
    """A 2-component system on [0,1], zero Dirichlet on both components:

    -u0'' + u1 = pi^2 sin(pi x) + sin(2 pi x)      ->  u0* = sin(pi x)
    -u1'' + u0 = 4 pi^2 sin(2 pi x) + sin(pi x)    ->  u1* = sin(2 pi x)
    """
    d = _line(ms)
    u, phi = d.fem_symbols(value_shape=(2,), order=order)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    s1, s2 = jno.np.sin(np.pi * xi), jno.np.sin(2 * np.pi * xi)
    f0 = (np.pi**2) * s1 + s2
    f1 = (4 * np.pi**2) * s2 + s1
    weak = jno.np.inner(ui.x, vi.x) + ui[1] * vi[0] + ui[0] * vi[1] - f0 * vi[0] - f1 * vi[1]
    return d, jno.fem([weak, u(xb)[0] - 0.0, u(xb)[1] - 0.0])


@pytest.mark.parametrize("order,tol", [(1, 1e-4), (2, 1e-6)])
def test_vector_unknown_recovers_manufactured(order, tol):
    """A vector unknown on a line. 1D refused ``vec>1`` outright, which ruled out every 1D *system*
    written as one field — a two-species model, a Timoshenko pair, a bar with several dofs per node.
    The element kernels were already written in terms of ``vec``; only the guard stood in the way.

    DOFs stay node-major (``dof = node*vec + comp``), which is what keeps ``fem.points`` a per-node
    array in 1D exactly as in 2D/3D."""
    d, fem = _vector_system(0.02, order)
    pts = np.asarray(fem.points).reshape(-1)
    assert fem.dofs == 2 * len(pts)
    sol = np.asarray(fem.solve()).reshape(len(pts), 2)
    assert np.max(np.abs(sol[:, 0] - np.sin(np.pi * pts))) < tol
    assert np.max(np.abs(sol[:, 1] - np.sin(2 * np.pi * pts))) < tol


def test_vector_unknown_stays_sparse():
    """The per-element scatter must survive ``vec>1``: an element couples its ``2*vec`` dofs, so nnz
    stays linear in the dof count rather than becoming a dense ``(2N)^2`` block."""
    d, fem = _vector_system(0.005)
    A = fem.operator[0]
    n = A.shape[0]
    assert hasattr(A, "indices"), "a vector 1D system must still assemble sparsely"
    assert int(A.nse) < 20 * n, f"nnz={int(A.nse)} is not linear in n={n}"


def test_vector_unknown_per_component_dirichlet():
    """Per-component essential conditions (``u(region)[i] - g``) address one component's dof stripe:
    pinning ``u0`` at both ends while ``u1`` keeps a natural condition on the right must give
    ``u0 = 2x`` exactly and leave ``u1`` free there."""
    d = _line(0.1)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([jno.np.inner(ui.x, vi.x) - 1.0 * vi[1], u(xl)[0] - 0.0, u(xl)[1] - 0.0, u(xr)[0] - 2.0])
    pts = np.asarray(fem.points).reshape(-1)
    sol = np.asarray(fem.solve()).reshape(len(pts), 2)
    assert np.max(np.abs(sol[:, 0] - 2.0 * pts)) < 1e-12  # -u0''=0, u0(0)=0, u0(1)=2
    # -u1''=1, u1(0)=0, natural u1'(1)=0 -> u1 = x - x^2/2, so u1(1)=1/2 (not pinned)
    assert abs(sol[int(np.argmax(pts)), 1] - 0.5) < 1e-10


def test_vector_unknown_three_components_independent_loads():
    """Extremes: three components with different loads must each recover their own solution — a
    component-blind scatter would give them all the same field."""
    d = _line(0.05)
    u, phi = d.fem_symbols(value_shape=(3,))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem(
        [
            jno.np.inner(ui.x, vi.x) - 1.0 * vi[0] - 2.0 * vi[1] + 3.0 * vi[2],
            u(xb)[0] - 0.0,
            u(xb)[1] - 0.0,
            u(xb)[2] - 0.0,
        ]
    )
    pts = np.asarray(fem.points).reshape(-1)
    sol = np.asarray(fem.solve()).reshape(len(pts), 3)
    for c, amp in enumerate((1.0, 2.0, -3.0)):  # -u'' = amp -> u = amp x(1-x)/2
        assert np.max(np.abs(sol[:, c] - amp * pts * (1 - pts) / 2)) < 1e-10, f"component {c}"


def test_vector_unknown_transient_seeds_every_component():
    """A vector field is given one initial condition **per component**, and each renders as a
    full-length vector that is zero outside its own stripe. This path used to read only
    ``ic_residuals[0]``, silently leaving every other component at zero — visible here because the two
    components decay at different rates and the second one would otherwise start (and stay) at zero."""
    d = _line(0.02, time=(0.0, 0.02, 21))
    u, phi = d.fem_symbols(value_shape=(2,))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[-1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    fem = jno.fem(
        [
            jno.np.inner(ui.t, vi) + jno.np.inner(ui.x, vi.x),
            u(xb)[0] - 0.0,
            u(xb)[1] - 0.0,
            u(ci)[0] - jno.np.sin(np.pi * ci),
            u(ci)[1] - jno.np.sin(2 * np.pi * ci),
        ]
    )
    x = np.asarray(fem.points).reshape(-1)
    s0 = np.asarray(fem.state0).reshape(len(x), 2)
    assert np.max(np.abs(s0[:, 0] - np.sin(np.pi * x))) < 1e-12
    assert np.max(np.abs(s0[:, 1] - np.sin(2 * np.pi * x))) < 1e-12, "the second component's IC was dropped"

    traj = np.asarray(fem.solve().fn())
    last = traj[-1].reshape(len(x), 2)
    t1 = float(fem.t1)
    e0 = np.exp(-(np.pi**2) * t1) * np.sin(np.pi * x)
    e1 = np.exp(-((2 * np.pi) ** 2) * t1) * np.sin(2 * np.pi * x)
    assert np.linalg.norm(last[:, 0] - e0) / np.linalg.norm(e0) < 2e-2
    assert np.linalg.norm(last[:, 1] - e1) / np.linalg.norm(e1) < 5e-2
    # the two components really decayed at their OWN rates: the amplitude ratio is exp(-3 pi^2 t),
    # which a shared initial state (or a component-blind march) could not produce
    ratio = np.max(np.abs(last[:, 1])) / np.max(np.abs(last[:, 0]))
    assert abs(ratio - np.exp(-3 * np.pi**2 * t1)) < 0.05, f"decay ratio {ratio:.3f}"


def test_vector_field_in_a_coupled_1d_system_fails_loud():
    """Scope: the coupled *block* path is still scalar per field. A vector field inside a coupled 1D
    system refuses rather than mis-sizing the blocks."""
    d = _line(0.1)
    u, v = d.fem_symbols(names=("u", "v"), value_shape=(2,))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi_, qi = p.bind(x=xi), q.bind(x=xi)
    with pytest.raises(NotImplementedError, match="scalar fields"):
        jno.fem(
            [
                jno.np.inner(ui.x, vi.x) + pi_ * vi[0],
                pi_.x * qi.x - qi,
                u(xb)[0] - 0.0,
                u(xb)[1] - 0.0,
                p(xb) - 0.0,
            ]
        )


def _reaction_diffusion(ms, order):
    """``-u'' + u = f`` with ``u = sin(pi x)``, zero Dirichlet — returns ``(ndof, max nodal error)``.

    The reaction term is deliberate. For the pure ``-u'' = f`` problem 1D P1 is *nodally exact* (the
    discrete Green's function reproduces nodal values), so a nodal-error study there measures the
    quadrature rule and reports ~O(h^4) for P1 too — it cannot tell P1 and P2 apart. Adding ``u*phi``
    breaks that exactness, so the nodal error shows the true convergence order."""
    d = _line(ms)
    u, phi = d.fem_symbols(order=order)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = (np.pi**2 + 1.0) * jno.np.sin(np.pi * xi)
    fem = jno.fem([ui.x * vi.x + u * vi - f * vi, u(xb) - 0.0])
    sol = np.asarray(fem.solve())
    pts = np.asarray(fem.points).reshape(-1)
    return len(sol), float(np.max(np.abs(sol - np.sin(np.pi * pts))))


def test_p2_line3_dof_layout_and_points():
    """P2 (LINE3) adds one dof per element **midpoint**, laid out after all vertices — so a vertex dof
    keeps its mesh-node index (which is what lets the boundary/Dirichlet lookup stay P1-shaped), and
    ``fem.points`` reports the dof coordinates the solution vector actually lives on."""
    d = _line(0.25)
    n_vert = int(np.asarray(d.mesh.points).shape[0])
    n_elem = int(np.asarray(d.mesh.cells_dict["line"]).shape[0])
    u, phi = d.fem_symbols(order=2)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x - 1.0 * vi, u(xb) - 0.0])

    pts = np.asarray(fem.points).reshape(-1)
    assert len(pts) == n_vert + n_elem, f"P2 needs one midpoint dof per element: {len(pts)} vs {n_vert + n_elem}"
    verts = np.asarray(d.mesh.points)[:, 0]
    assert np.allclose(pts[:n_vert], verts), "vertex dofs must come first, at their mesh-node index"
    cells = np.asarray(d.mesh.cells_dict["line"])
    assert np.allclose(np.sort(pts[n_vert:]), np.sort(0.5 * (verts[cells[:, 0]] + verts[cells[:, 1]])))
    assert fem.operator[0].shape == (len(pts), len(pts))

    # and it still solves: -u'' = 1, u(0)=u(1)=0 -> u = x(1-x)/2, which P2 represents EXACTLY
    sol = np.asarray(fem.solve())
    assert np.max(np.abs(sol - pts * (1.0 - pts) / 2.0)) < 1e-12, "P2 must be exact on a quadratic solution"


def test_p2_converges_faster_than_p1():
    """The point of LINE3: P1 is O(h²) at the nodes, P2 is O(h⁴) (nodal superconvergence, O(h^2k)),
    so at *equal dof count* P2 is orders more accurate — measured 4.7e-5 vs 2.4e-7 at 41 dofs."""
    p1 = [_reaction_diffusion(ms, 1) for ms in (0.1, 0.05, 0.025)]
    p2 = [_reaction_diffusion(ms, 2) for ms in (0.1, 0.05, 0.025)]

    r1 = [np.log2(p1[i][1] / p1[i + 1][1]) for i in range(len(p1) - 1)]
    r2 = [np.log2(p2[i][1] / p2[i + 1][1]) for i in range(len(p2) - 1)]
    assert all(abs(r - 2.0) < 0.25 for r in r1), f"P1 must converge at O(h^2), got rates {r1}"
    assert all(abs(r - 4.0) < 0.35 for r in r2), f"P2 must converge at O(h^4) nodally, got rates {r2}"

    # same dof count (41), far better answer -- the accuracy is not bought with dofs
    (n1, e1), (n2, e2) = p1[2], p2[1]
    assert n1 == n2 == 41
    assert e2 < e1 / 50.0, f"at {n1} dofs P2 ({e2:.2e}) must beat P1 ({e1:.2e}) by a wide margin"


def test_p2_is_sparse_and_couples_three_dofs_per_element():
    """A LINE3 element couples its 3 dofs, so the element block is 3x3 and ``nnz`` stays linear in the
    dof count — the sparse element scatter must not have quietly become dense for the higher order."""
    d = _line(0.01)
    u, phi = d.fem_symbols(order=2)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    A = jno.fem([ui.x * vi.x - 1.0 * vi, u(xb) - 0.0]).operator[0]
    n = A.shape[0]
    assert hasattr(A, "indices"), "P2 must assemble sparsely too"
    # ~9 entries per element (3x3) + 2 Dirichlet rows, against ~2 dofs per element
    assert int(A.nse) < 10 * n, f"nnz={int(A.nse)} is not linear in n={n}"


def test_p2_transient_decays_to_analytic():
    """P2 composes with the transient path: the mass, operator AND the initial state all live on the
    LINE3 dofs. The initial condition is the sharp edge — it is sampled at the dof coordinates, so a
    P2 state must be seeded at the midpoints too, not just at the mesh vertices."""
    d = _line(0.05, time=(0.0, 0.02, 41))
    u, phi = d.fem_symbols(order=2)
    sp = d.variable("interior", split=True)
    xi, ti = sp[0], sp[-1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x, u(xb) - 0.0, u(ci) - jno.np.sin(np.pi * ci)])

    pts = np.asarray(fem.points).reshape(-1)
    n_vert = int(np.asarray(d.mesh.points).shape[0])
    assert len(pts) > n_vert, "the transient state must be sized by the P2 dofs, not the vertices"

    traj = np.asarray(fem.solve().fn())
    assert traj.shape[1] == len(pts)
    # the IC must be seeded on every dof, midpoints included
    assert np.max(np.abs(traj[0] - np.sin(np.pi * pts))) < 1e-12, "P2 initial state is not on the dof nodes"
    exact = np.exp(-(np.pi**2) * float(fem.t1)) * np.sin(np.pi * pts)
    assert np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact) < 5e-3


_DUMMY_DOM = None


def _dummy_domain():
    global _DUMMY_DOM
    if _DUMMY_DOM is None:
        _DUMMY_DOM = jno.domain.from_array({"_": np.zeros((1, 1))})
    return _DUMMY_DOM


def _parametric_poisson(kappa, rhs_scale=1.0, ms=0.05, order=1):
    """``-kappa u'' = rhs_scale * pi^2 sin(pi x)``, zero Dirichlet. ``kappa`` may be a float or a
    ``jno.np.parameter`` (scalar or nodal field)."""
    d = _line(ms)
    u, phi = d.fem_symbols(order=order)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = rhs_scale * (np.pi**2) * jno.np.sin(np.pi * xi)
    return jno.fem([kappa * (ui.x * vi.x) - f * vi, u(xb) - 0.0])


def test_scalar_runtime_parameter_makes_the_system_parametric():
    """A ``jno.np.parameter`` in a 1D form makes the system parametric — the differentiable-inverse
    entry. This previously did not merely fail, it failed *cryptically*: the 1D assembler threaded no
    runtime parameters at all, so the value never reached the kernel and it died with an internal
    ``KeyError`` about ``InternalVars`` rather than any documented error."""
    from jno.trace import FemLinearSystem

    k = jno.np.parameter((1,), name="k")
    k.initialize(jax.nn.initializers.constant(2.5))
    fem = _parametric_poisson(k)
    assert isinstance(fem.operator, FemLinearSystem) and fem.operator.is_parametric
    assert list(fem.operator.runtime_parameter_exprs) == ["k"]

    node = fem.solve()
    assert not isinstance(node, jax.Array), "a parametric 1D solve must be a trace node, not an array"
    # evaluated at k, it must equal the non-parametric assembly at the same value
    crux = jno.core([(node * 0.0).mae], domain=_dummy_domain())
    got = np.asarray(crux.eval([node])).reshape(-1)
    ref = np.asarray(_parametric_poisson(2.5).solve()).reshape(-1)
    assert np.max(np.abs(got - ref)) < 1e-10, "parametric 1D solve disagrees with the constant-coefficient one"


def test_1d_inverse_recovers_a_scalar_parameter():
    """End to end: recover a diffusivity from full-field 1D data through ``crux.solve``. This is what
    the missing parameter path cost — a *differentiable* library in which the cheapest, most natural
    dimension for prototyping an inverse problem could not run one."""
    import optax

    k_true = 2.5
    u_obs = np.asarray(_parametric_poisson(k_true, rhs_scale=k_true).solve()).reshape(-1)

    k = jno.np.parameter((1,), name="k")
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(0.15))
    node = _parametric_poisson(k, rhs_scale=k_true).solve()
    crux = jno.core([(node - u_obs).mae], domain=_dummy_domain())
    crux.solve(220)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - k_true) < 0.05, f"kappa not recovered through the 1D inverse: {rec:.4f} vs {k_true}"


def test_nodal_field_parameter_in_1d():
    """A nodal FIELD parameter ``k(x)`` rides the same path: it is gathered per element and interpolated
    with that element's shape functions. A *constant* field must therefore reproduce the equivalent
    scalar-coefficient solve exactly — which is the check that the gather and interpolation line up."""
    d = _line(0.05)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    kf = jno.np.parameter(phi, name="kf")
    kf.initialize(jax.nn.initializers.constant(2.0))
    fem = jno.fem([kf * (ui.x * vi.x) - 1.0 * vi, u(xb) - 0.0])

    crux = jno.core([(fem.solve() * 0.0).mae], domain=_dummy_domain())
    got = np.asarray(crux.eval([fem.solve()])).reshape(-1)
    x = np.asarray(fem.points).reshape(-1)
    assert np.max(np.abs(got - x * (1.0 - x) / 4.0)) < 1e-12, "constant field k=2 must give u = x(1-x)/4"


def test_1d_nonlinear_inverse_recovers_a_parameter():
    """A parameter in a NONLINEAR 1D form. The residual already re-evaluates its coefficients from
    ``args`` — which is what made the linear path parametric — and ``FemResidualOperator`` takes
    ``R(u, args)``, so Newton runs on ``R(., theta)`` and the implicit derivative gives d(u)/d(theta)
    with no extra machinery. Here ``-k u'' + u^3 = f``, recovered from a 0.6 start."""
    import optax

    from jno.trace import FemResidualOperator

    k_true = 1.6

    def build(kappa):
        d = _line(0.08)
        u, phi = d.fem_symbols()
        xi = d.variable("interior", split=True)[0]
        xb = d.variable("boundary", split=True)[0]
        ui, vi = u.bind(x=xi), phi.bind(x=xi)
        return jno.fem([kappa * (ui.x * vi.x) + (u * u * u) * vi - 8.0 * vi, u(xb) - 0.0])

    u_obs = np.asarray(build(k_true).solve()).reshape(-1)
    k = jno.np.parameter((1,), name="kn")
    k.initialize(jax.nn.initializers.constant(0.6))
    k.optimizer(optax.adam(0.1))
    femp = build(k)
    assert isinstance(femp.operator, FemResidualOperator)
    assert list(femp.operator.runtime_parameter_exprs) == ["kn"]

    crux = jno.core([(femp.solve() - u_obs).mae], domain=_dummy_domain())
    crux.solve(200)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - k_true) < 0.08, f"kappa not recovered through the nonlinear 1D inverse: {rec:.4f}"


def test_1d_parameter_scope_limits_fail_loud():
    """A field parameter must share the trial's nodal layout, so a P1 field cannot ride a LINE3
    element. That refusal comes from ``jno.np.parameter`` itself, at construction, which is earlier
    and better than assembly. (The coupled-system limit this test also used to cover is gone: a
    coupled STEADY 1D form is now parametric — see the coupled section. The coupled TRANSIENT block
    is still refused, covered by ``test_coupled_1d_transient_parameter_fails_loud``.)"""
    d2 = _line(0.1)
    _u2, phi2 = d2.fem_symbols(order=2)
    with pytest.raises(NotImplementedError, match="P1|order=1"):
        jno.np.parameter(phi2, name="kf2")


def _const_net_1d(c):
    """A 'network' emitting a constant per quad point — the degenerate case that must reproduce a
    scalar-coefficient assembly *exactly*, which is what pins the kernel's evaluation of the net."""
    eqx = pytest.importorskip("equinox", reason="neural coefficients need equinox")

    class _Const(eqx.Module):
        c: jnp.ndarray

        def __call__(self, *args):
            n = jnp.asarray(args[0]).shape[0]
            return jnp.broadcast_to(self.c.reshape(1, 1), (n, 1))

    net = jno.nn.wrap(_Const(c=jnp.asarray(float(c), dtype=jnp.float64)))
    net.dtype(jnp.float64)
    return net


def _neural_poisson(kfun, rhs_scale=1.0, ms=0.05):
    """``-k(x) u'' = rhs_scale * pi^2 sin(pi x)``, zero Dirichlet, with ``k`` from a callable."""
    d = _line(ms)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = rhs_scale * (np.pi**2) * jno.np.sin(np.pi * xi)
    return jno.fem([kfun(xi) * (ui.x * vi.x) - f * vi, u(xb) - 0.0])


def test_neural_coefficient_in_1d_matches_the_scalar_assembly():
    """A ``jno.nn.wrap`` coefficient assembles on a 1D domain. Unlike a scalar/field parameter a network
    never enters the per-cell ``volume_vars`` — a weight pytree is cell-independent — so the kernel
    re-evaluates it at the quad points from a ``{name: module}`` table instead, and its weights ride
    ``args`` as a ``ModelWeights`` slot. A *constant* net must therefore land on the scalar-coefficient
    solve exactly; anything else means the net is being evaluated on the wrong points."""
    from jno.trace import FemLinearSystem, ModelWeights

    net = _const_net_1d(0.8)
    fem = _neural_poisson(lambda xi: net(xi))
    assert isinstance(fem.operator, FemLinearSystem) and fem.operator.is_parametric
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)

    node = fem.solve()
    assert not isinstance(node, jax.Array), "a neural 1D solve must be a trace node"
    crux = jno.core([(node * 0.0).mae], domain=_dummy_domain())
    got = np.asarray(crux.eval([node])).reshape(-1)
    ref = np.asarray(_neural_poisson(lambda xi: 0.8).solve()).reshape(-1)
    assert np.max(np.abs(got - ref)) < 1e-12, "a constant net must reproduce the scalar-coefficient solve"


def test_1d_neural_coefficient_trains():
    """The point of the layer: ``d(solve)/d(weights)`` flows, so a coefficient network is *learnable*
    from 1D data — the differentiable-FEM-plus-ML story working in the dimension you would prototype
    it in. Starts a 4x-wrong coefficient and drives the data misfit down through ``crux.solve``."""
    optax = pytest.importorskip("optax")

    k_true = 2.0
    u_obs = np.asarray(_neural_poisson(lambda xi: k_true, rhs_scale=k_true).solve()).reshape(-1)

    net = _const_net_1d(0.5)  # 4x off
    net.optimizer(optax.adam(0.1))
    node = _neural_poisson(lambda xi: net(xi), rhs_scale=k_true).solve()
    crux = jno.core([(node - u_obs).mae], domain=_dummy_domain())

    before = float(np.max(np.abs(np.asarray(crux.eval([node])).reshape(-1) - u_obs)))
    crux.solve(200)
    after = float(np.max(np.abs(np.asarray(crux.eval([node])).reshape(-1) - u_obs)))
    assert after < before / 10.0, f"training must reduce the 1D misfit: {before:.3e} -> {after:.3e}"
    assert after < 5e-2, f"the coefficient network did not fit the data: {after:.3e}"


def _transient_diffusion(kappa, ms=0.08):
    """``u_t = kappa u_xx`` on the unit line, ``u0 = sin(pi x)``, zero Dirichlet — analytic decay
    ``exp(-kappa pi^2 t) sin(pi x)``. ``kappa`` may be a float or a ``jno.np.parameter``."""
    d = _line(ms, time=(0.0, 0.05, 21))
    u, phi = d.fem_symbols()
    sp = d.variable("interior", split=True)
    xi, ti = sp[0], sp[-1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    return jno.fem([ui.t * vi + kappa * (ui.x * vi.x), u(xb) - 0.0, u(ci) - jno.np.sin(np.pi * ci)])


def test_1d_transient_inverse_recovers_a_diffusivity():
    """The canonical PDE inverse problem — recover a diffusivity from a time series — now runs in 1D.

    The transient operator and load re-form from the runtime args at every step, so ``∂traj/∂θ`` flows
    through the marcher. Starts nearly 3x wrong (2.0 against a true 0.7)."""
    import optax

    k_true = 0.7
    fem = _transient_diffusion(k_true)
    traj_obs = np.asarray(fem.solve().fn())
    pts = np.asarray(fem.points).reshape(-1)
    exact = np.exp(-k_true * np.pi**2 * float(fem.t1)) * np.sin(np.pi * pts)
    assert np.linalg.norm(traj_obs[-1] - exact) / np.linalg.norm(exact) < 5e-3, "forward transient is off"

    k = jno.np.parameter((1,), name="kt")
    k.initialize(jax.nn.initializers.constant(2.0))
    k.optimizer(optax.adam(0.08))
    femp = _transient_diffusion(k)
    assert list(femp.operator.runtime_parameter_exprs) == ["kt"], "the transient block must carry the parameter"

    node = femp.solve()
    crux = jno.core([(node - traj_obs).mae], domain=_dummy_domain())
    crux.solve(200)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - k_true) < 0.03, f"diffusivity not recovered from the 1D time series: {rec:.4f} vs {k_true}"


def test_1d_transient_parameter_on_the_mass_fails_loud():
    """A parameter on the MASS term (``u_t * phi``) must fail loud, not be silently frozen.

    The mass is assembled once, outside the per-args re-forming, so a parameter there would be read at
    its zero placeholder and baked in — a wrong answer with no error, which is the one outcome this
    stack never returns. The 2D/3D path documents the same rule; here it is enforced."""
    d = _line(0.1, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols()
    sp = d.variable("interior", split=True)
    xi, ti = sp[0], sp[-1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    rho = jno.np.parameter((1,), name="rho")
    rho.initialize(jax.nn.initializers.constant(1.0))
    with pytest.raises(NotImplementedError, match="MASS|mass"):
        jno.fem([rho * (ui.t * vi) + ui.x * vi.x, u(xb) - 0.0, u(ci) - jno.np.sin(np.pi * ci)])


# ==========================================================================
# higher-order Lagrange (P3, P4, …) on a line
# ==========================================================================
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_higher_order_dof_layout(order):
    """A degree-``k`` line carries ``k-1`` **interior** dofs per element, laid out after all vertices
    and element-major. 1D was capped at order 2 while 2D/3D had P3+; the cap is gone, and orders above
    2 are tabulated by basix on the reference interval through the very builder the 2D/3D path uses,
    so there is no second hand-written basis to drift.

    Vertices must stay first and at their mesh index — that is what keeps the boundary/Dirichlet node
    lookup order-agnostic (a 1D boundary is an endpoint, hence always a vertex)."""
    d = _line(0.25)
    n_vert = int(np.asarray(d.mesh.points).shape[0])
    n_elem = int(np.asarray(d.mesh.cells_dict["line"]).shape[0])
    u, phi = d.fem_symbols(order=order)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x - 1.0 * vi, u(xb) - 0.0])

    pts = np.asarray(fem.points).reshape(-1)
    assert len(pts) == n_vert + n_elem * (order - 1)
    assert np.allclose(pts[:n_vert], np.asarray(d.mesh.points)[:, 0]), "vertex dofs must come first"
    assert fem.operator[0].shape == (len(pts), len(pts))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_higher_order_reproduces_its_own_degree_exactly(order):
    """The decisive check on a higher-order basis: P{k} contains every polynomial of degree k, so on a
    manufactured degree-k solution it must be exact to machine precision — a wrong basis, a scrambled
    dof order or a misplaced interior node all fail this while still converging plausibly.

    ``-u'' = -k(k-1) x^(k-2)`` with ``u(0)=0``, ``u(1)=1`` has the exact solution ``u = x^k``."""
    d = _line(0.25)
    u, phi = d.fem_symbols(order=order)
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = -float(order * (order - 1)) * (xi ** (order - 2)) if order > 2 else -2.0 + 0.0 * xi
    fem = jno.fem([ui.x * vi.x - f * vi, u(xl) - 0.0, u(xr) - 1.0])
    pts = np.asarray(fem.points).reshape(-1)
    sol = np.asarray(fem.solve()).reshape(-1)
    assert np.max(np.abs(sol - pts**order)) < 1e-12, f"P{order} is not exact on x^{order}"


def test_p3_vertex_superconvergence_beats_p2():
    """P3 must actually converge like a cubic. Measured at the **vertices**, where 1D Lagrange is
    superconvergent at O(h^2k): P2 gives ~4, P3 gives ~6. (Over *all* dofs the interior nodes pull the
    rate down to the interpolation order, so the two metrics must not be mixed.)

    The reaction term is essential: for pure ``-u'' = f`` 1D Lagrange is nodally exact at every order,
    so a nodal-error study there measures the quadrature rule and reports the same rate for P1 and P4
    alike."""

    def vertex_err(ms, order):
        d = _line(ms)
        u, phi = d.fem_symbols(order=order)
        xi = d.variable("interior", split=True)[0]
        xb = d.variable("boundary", split=True)[0]
        ui, vi = u.bind(x=xi), phi.bind(x=xi)
        f = (np.pi**2 + 1.0) * jno.np.sin(np.pi * xi)
        fem = jno.fem([ui.x * vi.x + ui * vi - f * vi, u(xb) - 0.0])
        sol = np.asarray(fem.solve()).reshape(-1)
        pts = np.asarray(fem.points).reshape(-1)
        n_vert = int(np.asarray(d.mesh.points).shape[0])
        return float(np.max(np.abs(sol[:n_vert] - np.sin(np.pi * pts[:n_vert]))))

    sizes = (0.2, 0.1, 0.05)
    for order, expected in ((2, 4.0), (3, 6.0)):
        errs = [vertex_err(ms, order) for ms in sizes]
        rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert all(abs(r - expected) < 0.4 for r in rates), f"P{order} vertex rates {rates}, expected ~{expected}"


def test_higher_order_stays_sparse():
    """A P3 element couples its 4 dofs, so nnz stays linear in the dof count — the element scatter must
    not have quietly become dense for the higher order."""
    d = _line(0.005)
    u, phi = d.fem_symbols(order=3)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    A = jno.fem([ui.x * vi.x - vi, u(xb) - 0.0]).operator[0]
    n = A.shape[0]
    assert hasattr(A, "indices")
    assert int(A.nse) < 12 * n, f"nnz={int(A.nse)} is not linear in n={n}"


def test_order_below_1_fails_loud():
    """Scope: a Lagrange element needs order >= 1."""
    from jno.utils.solver.fem_1d import _line_shape

    with pytest.raises(NotImplementedError, match="order >= 1"):
        _line_shape(np.asarray([0.5]), 0)


# ==========================================================================
# steady scalar — linear, all BC kinds, recovered exactly
# ==========================================================================
def test_poisson_dirichlet_recovers_linear():
    # -u'' = 0, u(0)=0, u(1)=1 -> u = x (LINE2-exact).
    d = _line(0.1)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, u(xl) - 0.0, u(xr) - 1.0])
    assert fem.is_linear
    sol = _solve(fem)
    c = _x(d)
    assert np.linalg.norm(sol - c) / np.linalg.norm(c) < 1e-9
    A = _dense(fem.A)
    assert np.allclose(A, A.T, atol=1e-12)  # symmetric (Dirichlet elimination keeps it so)


def test_poisson_dirichlet_bubble_nodally_exact():
    # -u'' = 2, u(0)=u(1)=0 -> u = x(1-x). 1D P1 FEM is nodally exact for -u''=f.
    d = _line(0.05)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x - 2.0 * vi, u(xl) - 0.0, u(xr) - 0.0])
    sol = _solve(fem)
    c = _x(d)
    exact = c * (1 - c)
    assert np.linalg.norm(sol - exact) / np.linalg.norm(exact) < 1e-9


def test_poisson_neumann_recovers_linear():
    # -u'' = 0, u(0)=0, du/dn=1 on the right endpoint -> u = x. Boundary term -g*phi.
    d = _line(0.1)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, -1.0 * phi.bind(x=xr), u(xl) - 0.0])
    assert "surface@right" in fem.classification
    sol = _solve(fem)
    assert np.linalg.norm(sol - _x(d)) / np.linalg.norm(_x(d)) < 1e-8


def test_poisson_robin_recovers_linear():
    # du/dn + a u = 1 + a on the right endpoint, u(0)=0 -> u = x. The a*u term must
    # land in the matrix (unified boundary path), not just the load.
    a = 2.0
    d = _line(0.1)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    robin = (a * u.bind(x=xr) - (1.0 + a)) * phi.bind(x=xr)
    fem = jno.fem([ui.x * vi.x, robin, u(xl) - 0.0])
    assert "surface@right" in fem.classification
    sol = _solve(fem)
    assert np.linalg.norm(sol - _x(d)) / np.linalg.norm(_x(d)) < 1e-8


# ==========================================================================
# steady nonlinear
# ==========================================================================
def test_nonlinear_reaction_newton_recovers_manufactured():
    spo = pytest.importorskip("scipy.optimize")
    # -u'' + u^3 = f, u_exact = x(1-x), f = 2 + (x(1-x))^3, u(0)=u(1)=0.
    d = _line(0.02)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = 2.0 + (xi * (1 - xi)) ** 3
    fem = jno.fem([ui.x * vi.x + (u * u * u) * vi - f * vi, u(xl) - 0.0, u(xr) - 0.0])
    assert not fem.is_linear
    sol = spo.root(
        lambda v: np.asarray(fem.residual(v)),
        np.zeros(fem.dofs),
        jac=lambda v: _dense(fem.jacobian(v)),
        method="hybr",
    )
    assert sol.success
    c = _x(d)
    exact = c * (1 - c)
    assert np.linalg.norm(sol.x - exact) / np.linalg.norm(exact) < 1e-2


# ==========================================================================
# transient (1D + time)
# ==========================================================================
def test_transient_heat_decays_to_analytic():
    nu = 1.0
    d = _line(0.02, time=(0.0, 0.02, 21))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    u, phi = d.fem_symbols()
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    ic = u(ci[0]) - jno.fn(lambda x: jnp.sin(jnp.pi * x), [ci[0]])
    fem = jno.fem([ui.t * vi + nu * (ui.x * vi.x), u(xb) - 0.0, ic])
    assert fem.is_transient and fem.is_linear

    M, A = _dense(fem.M), _dense(fem.operator.A)
    assert np.allclose(M, M.T) and np.allclose(A, A.T)
    w, dt = np.asarray(fem.state0), float(fem.dt)
    nsteps = round((fem.t1 - fem.t0) / dt)
    for _ in range(nsteps):  # backward Euler
        w = np.linalg.solve(M + dt * A, M @ w)

    c = _x(d)
    analytic = np.exp(-nu * np.pi**2 * fem.t1) * np.sin(np.pi * c)
    assert np.linalg.norm(w - analytic) / np.linalg.norm(analytic) < 1e-2
    assert 0.0 < np.linalg.norm(w) < np.linalg.norm(np.asarray(fem.state0))  # decays


def test_transient_nonlinear_assembles_residual_block():
    # 1D Allen-Cahn-style reaction: u_t*phi + u_x*phi_x + (u^3 - u)*phi.
    d = _line(0.05, time=(0.0, 0.1, 6))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    u, phi = d.fem_symbols()
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    fem = jno.fem([ui.t * vi + (ui.x * vi.x) + (u * u * u - u) * vi, u(xb) - 0.0, u(ci[0]) - 0.0])
    assert fem.is_transient and not fem.is_linear
    block = fem.operator
    assert block.residual is not None and block.jacobian is not None and block.mass is not None
    R0 = np.asarray(block.residual(np.asarray(fem.state0), float(fem.t0), None))
    assert R0.shape == (fem.dofs,)


# ==========================================================================
# coupled / mixed multi-field 1D (native block assembly)
# ==========================================================================
# coupled 1D is assembled by a dedicated block residual
# (jno/utils/solver/fem_1d.py::assemble_fem_1d_multifield). There is no separate problem
# object here (fem.problem is None), so the block layout is hand-computed: field i occupies
# sol[i*n : (i+1)*n] for scalar fields. The manufactured pairs use ASYMMETRIC cross-
# coupling so a transposed/mis-scattered block would change the solution.
def test_coupled_linear_recovers():
    # -u'' + p = 2x ; -p'' + 3u = 3x ; u=x, p=2x on the boundary (asymmetric: the p->u
    # coupling coeff is 1, the u->p coeff is 3). u*=x, p*=2x are LINE2-exact.
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem(
        [
            ui.x * vi.x + 1.0 * pi * vi - 2.0 * xi * vi,
            pi.x * qi.x + 3.0 * ui * qi - 3.0 * xi * qi,
            u(xb) - xb,
            p(xb) - 2.0 * xb,
        ]
    )
    assert fem.is_linear and fem.dofs == 2 * n
    A = _dense(fem.A)
    # off-diagonal blocks present and DIFFERENT (asymmetric coupling -> not transposed)
    assert np.any(np.abs(A[:n, n:]) > 1e-12) and np.any(np.abs(A[n:, :n]) > 1e-12)
    assert not np.allclose(A[:n, n:], A[n:, :n])
    sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    c = _x(d)
    assert np.linalg.norm(sol[:n] - c) / np.linalg.norm(c) < 1e-9  # u = x
    assert np.linalg.norm(sol[n:] - 2 * c) / np.linalg.norm(2 * c) < 1e-9  # p = 2x


def test_coupled_nonlinear_recovers():
    # Nonlinear coupled: -u'' + u*p = 2x^2 ; -p'' + u^2 = x^2 ; u=x, p=2x. u*=x, p*=2x
    # solve it (the nonlinear terms equal their sources at the solution); Newton recovers.
    spo = pytest.importorskip("scipy.optimize")
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem(
        [
            ui.x * vi.x + (u * p) * vi - 2.0 * (xi * xi) * vi,
            pi.x * qi.x + (u * u) * qi - (xi * xi) * qi,
            u(xb) - xb,
            p(xb) - 2.0 * xb,
        ]
    )
    assert not fem.is_linear and fem.dofs == 2 * n
    sol = spo.root(
        lambda w: np.asarray(fem.residual(w)),
        np.zeros(fem.dofs),
        jac=lambda w: _dense(fem.jacobian(w)),
        method="hybr",
    )
    assert sol.success
    c = _x(d)
    assert np.linalg.norm(sol.x[:n] - c) / np.linalg.norm(c) < 1e-7
    assert np.linalg.norm(sol.x[n:] - 2 * c) / np.linalg.norm(2 * c) < 1e-7


def test_coupled_mixed_bc_recovers():
    # Coupled with mixed BCs: u is Dirichlet at the left and Neumann at the right
    # (du/dn = 1), p is Dirichlet at both ends. -u'' + p = 2x, -p'' + u = x; u*=x, p*=2x.
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    vr = v.bind(x=xr)
    fem = jno.fem(
        [
            ui.x * vi.x + pi * vi - 2.0 * xi * vi,
            pi.x * qi.x + ui * qi - xi * qi,
            u(xl) - 0.0,  # Dirichlet (left)
            -1.0 * vr,  # Neumann du/dn = 1 (right)
            p(xb) - 2.0 * xb,  # Dirichlet p = 2x (both ends)
        ]
    )
    assert "surface@right" in fem.classification and "dirichlet@left" in fem.classification
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    c = _x(d)
    assert np.linalg.norm(sol[:n] - c) / np.linalg.norm(c) < 1e-9
    assert np.linalg.norm(sol[n:] - 2 * c) / np.linalg.norm(2 * c) < 1e-9


def test_coupled_transient_decays_to_analytic():
    # Coupled 1D transient, asymmetric: u_t = u'' (heat), p_t = p'' + c*u (driven by u).
    # IC u0 = sin(pi x), p0 = 0; u = e^{-pi^2 t} sin(pi x), p = c*t*e^{-pi^2 t} sin(pi x).
    cc = 4.0
    d = _line(0.02, time=(0.0, 0.05, 51))
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x,
            pi.t * qi + pi.x * qi.x - cc * u.bind(x=xi, t=ti) * qi,
            u(xb) - 0.0,
            p(xb) - 0.0,
            u(ci) - jno.fn(lambda x: jnp.sin(jnp.pi * x), [ci]),
            p(ci) - 0.0,
        ]
    )
    assert fem.is_transient and fem.is_linear and fem.dofs == 2 * n
    M, A = _dense(fem.M), _dense(fem.operator.A)
    assert np.allclose(M[:n, n:], 0.0)  # mass block-diagonal
    w = np.asarray(fem.state0).copy()
    assert np.allclose(w[n:], 0.0)  # p starts at 0
    dt = float(fem.dt)
    for _ in range(round((fem.t1 - fem.t0) / dt)):  # backward Euler
        w = np.linalg.solve(M + dt * A, M @ w)
    cx = _x(d)
    decay = np.exp(-(np.pi**2) * fem.t1)
    u_ex, p_ex = decay * np.sin(np.pi * cx), cc * fem.t1 * decay * np.sin(np.pi * cx)
    assert np.linalg.norm(w[:n] - u_ex) / np.linalg.norm(u_ex) < 1e-2
    assert np.linalg.norm(w[n:] - p_ex) / np.linalg.norm(p_ex) < 2e-2
    assert np.linalg.norm(w[n:]) > 1e-3  # p grew from zero via the coupling


def test_coupled_nonlinear_transient_recovers():
    # The full triple in 1D: nonlinear + coupled + transient. Zero-flux (natural Neumann),
    # spatially-uniform: u_t = -u, p_t = u^2 (the u^2 makes it nonlinear), u(0)=1, p(0)=0
    # -> u = e^{-t}, p = (1 - e^{-2t})/2. Newton backward-Euler recovers it.
    d = _line(0.1, time=(0.0, 0.1, 11))
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    uu = u.bind(x=xi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui * vi, pi.t * qi + pi.x * qi.x - (uu * uu) * qi, u(ci) - 1.0, p(ci) - 0.0])
    assert fem.is_transient and not fem.is_linear and fem.dofs == 2 * n
    op = fem.operator
    M = _dense(op.mass(0.0, None))
    w = np.asarray(fem.state0).copy()
    dt = float(fem.dt)
    for _ in range(round((fem.t1 - fem.t0) / dt)):  # Newton backward-Euler
        w_old = w.copy()
        for _ in range(30):
            G = M @ (w - w_old) / dt + np.asarray(op.residual(w, 0.0, None))
            if np.linalg.norm(G) < 1e-11:
                break
            w = w - np.linalg.solve(M / dt + _dense(op.jacobian(w, 0.0, None)), G)
    u_ex = np.exp(-fem.t1)
    p_ex = (1.0 - np.exp(-2.0 * fem.t1)) / 2.0
    assert w[:n].std() < 1e-10 and w[n:].std() < 1e-10  # spatially uniform
    assert abs(w[:n].mean() - u_ex) / u_ex < 1e-2
    assert abs(w[n:].mean() - p_ex) / p_ex < 3e-2


# ==========================================================================
# coupled 1D — second order in time (u_tt)
# ==========================================================================
def test_coupled_1d_second_order_two_waves_beat():
    """Two coupled 1D waves, ``u_tt`` on both fields. 1D refused this outright ("single-field only"),
    though the 2D/3D coupled route builds exactly this augmented block.

        u_tt = u'' - k(u - p),   p_tt = p'' - k(p - u),   clamped, u(x,0)=sin(pi x), p(x,0)=0, at rest

    The normal modes decouple: ``s = u+p`` gives ``omega_s = pi``; ``a = u-p`` gives
    ``omega_a = sqrt(pi^2 + 2k)``. With this IC ``s0 = a0 = sin(pi x)``, so

        u = 0.5 sin(pi x) [cos(omega_s t) + cos(omega_a t)]
        p = 0.5 sin(pi x) [cos(omega_s t) - cos(omega_a t)]

    — energy sloshes between the two membranes at the beat frequency. That is the assertion a
    *decoupled* or frozen operator cannot pass: it would leave p at zero for all time."""
    k = 5.0
    d = _line(0.02, time=(0.0, 2.0, 200))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[-1]
    xb = d.variable("boundary", split=True)[0]
    ic = d.variable("initial", split=True)
    xi0, ti0 = ic[0], ic[-1]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi_, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    ui0, pi0 = u.bind(x=xi0, t=ti0), p.bind(x=xi0, t=ti0)
    fem = jno.fem(
        [
            ui.tt * vi + ui.x * vi.x + k * (ui - pi_) * vi,
            pi_.tt * qi + pi_.x * qi.x + k * (pi_ - ui) * qi,
            u(xb) - 0.0,
            p(xb) - 0.0,
            u(xi0) - jno.fn(lambda x: jnp.sin(np.pi * x), [xi0]),
            p(xi0) - 0.0,
            ui0.t - 0.0,
            pi0.t - 0.0,
        ]
    )
    assert fem.is_transient and fem.is_linear
    # the augmented state is [u, p, u̇, ṗ]: displacement blocks then velocity blocks
    o = fem.offsets
    assert len(o) == 5 and o[2] == o[4] - o[2], f"augmented layout not [u, p | u̇, ṗ]: {o}"

    traj = np.asarray(fem.solve().fn())
    x = np.asarray(fem.field_points[0]).reshape(-1)
    ts = np.linspace(0.0, float(fem.t1), traj.shape[0])
    w = np.sin(np.pi * x)
    amp = lambda blk: (blk @ w) / (w @ w)  # modal amplitude of the sin(pi x) mode  # noqa: E731
    au = np.array([amp(traj[i, o[0] : o[1]]) for i in range(traj.shape[0])])
    ap = np.array([amp(traj[i, o[1] : o[2]]) for i in range(traj.shape[0])])
    ws, wa = np.pi, np.sqrt(np.pi**2 + 2 * k)
    au_ex = 0.5 * (np.cos(ws * ts) + np.cos(wa * ts))
    ap_ex = 0.5 * (np.cos(ws * ts) - np.cos(wa * ts))
    assert np.linalg.norm(au - au_ex) / np.linalg.norm(au_ex) < 5e-3
    assert np.linalg.norm(ap - ap_ex) / np.linalg.norm(ap_ex) < 5e-3
    # p started at rest and at zero: it can only move through the coupling
    assert np.max(np.abs(ap)) > 0.8


def test_coupled_1d_second_order_scope_limits_fail_loud():
    """The augmented ``[u_all; v_all]`` form carries one velocity block per displacement block, so a
    bare ``u_t`` (damping, or a first-order field) and a field with no inertia at all are both
    inexpressible — each refuses, naming the first-order-system rewrite. Same scope as 2D/3D."""
    d = _line(0.1, time=(0.0, 0.2, 21))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[-1]
    xb = d.variable("boundary", split=True)[0]
    ic = d.variable("initial", split=True)
    xi0, ti0 = ic[0], ic[-1]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi_, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    ui0, pi0 = u.bind(x=xi0, t=ti0), p.bind(x=xi0, t=ti0)
    common = [u(xb) - 0.0, p(xb) - 0.0, u(xi0) - 1.0, p(xi0) - 0.0, ui0.t - 0.0, pi0.t - 0.0]

    with pytest.raises(NotImplementedError, match="u_t term"):  # damping
        jno.fem([ui.tt * vi + ui.x * vi.x + 0.5 * ui.t * vi, pi_.tt * qi + pi_.x * qi.x, *common])

    with pytest.raises(NotImplementedError, match="EVERY field to carry u_tt"):  # p has no inertia
        jno.fem([ui.tt * vi + ui.x * vi.x + pi_ * vi, pi_.x * qi.x + ui * qi, *common])


# ==========================================================================
# coupled 1D — algebraic (DAE) fields in a transient block
# ==========================================================================
def test_coupled_1d_transient_algebraic_field():
    """A coupled transient field with **no** time derivative is algebraic: its rows of ``M`` are zero,
    so the block is a DAE and the implicit step solves ``A p = c`` on those rows. The 1D path used to
    require a temporal term on *every* field, which ruled out constraint/closure fields (a pressure, a
    saturation, an equilibrium concentration) that the 2D/3D coupled path has always accepted.

        u_t = u''      clamped, u(x,0) = sin(pi x)   ->  u = e^{-pi^2 t} sin(pi x)
        p   = c u      algebraic closure, no p_t     ->  p = c u at every step
    """
    c = 3.0
    d = _line(0.02, time=(0.0, 0.05, 51))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi_, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x,
            pi_ * qi - c * u.bind(x=xi, t=ti) * qi,  # algebraic: no p_t
            u(xb) - 0.0,
            p(xb) - 0.0,
            u(ci) - jno.np.sin(np.pi * ci),
        ]
    )
    assert fem.is_transient and fem.is_linear
    n = fem.offsets[1]

    # the structural signature of a DAE: the algebraic field contributes nothing to the mass
    M = _dense(fem.M)
    assert np.allclose(M[n:, :], 0.0), "an algebraic field must have zero mass rows"
    assert np.any(np.abs(M[:n, :n]) > 1e-12), "the evolving field must still have a mass"

    traj = np.asarray(fem.solve().fn())
    assert traj.shape[1] == fem.dofs
    x = np.asarray(fem.field_points[0]).reshape(-1)
    u_ex = np.exp(-(np.pi**2) * float(fem.t1)) * np.sin(np.pi * x)
    assert np.linalg.norm(traj[-1, :n] - u_ex) / np.linalg.norm(u_ex) < 1e-2
    # the constraint holds at the final step, and p is genuinely driven (not left at zero)
    assert np.max(np.abs(traj[-1, n:] - c * traj[-1, :n])) < 1e-6
    assert np.max(np.abs(traj[-1, n:])) > 1e-2


def test_coupled_1d_all_algebraic_is_not_a_transient_problem():
    """Dropping the per-field requirement must not make *every* field algebraic acceptable: with no
    time derivative anywhere there is nothing to march, and the initial condition has no meaning. It
    still errors (from the IC guard, which is the earlier and more specific one)."""
    d = _line(0.1, time=(0.0, 0.02, 11))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi_, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    with pytest.raises(ValueError, match="no time derivative"):
        jno.fem(
            [
                ui.x * vi.x + pi_ * vi,
                pi_ * qi - u.bind(x=xi, t=ti) * qi,
                u(xb) - 0.0,
                p(xb) - 0.0,
                u(ci) - 1.0,
            ]
        )


# ==========================================================================
# coupled 1D — per-field element order (P2 and mixed / Taylor-Hood shape)
# ==========================================================================
def _coupled_orders(ms, ou, op):
    """``-u'' + p = f1 ; -p'' + u = f2`` with ``u* = sin(pi x)``, ``p* = sin(2 pi x)``, zero Dirichlet.
    Each field is requested at its own Lagrange order."""
    d = _line(ms)
    u, v = d.fem_symbols(names=("u", "v"), order=ou)
    p, q = d.fem_symbols(names=("p", "q"), order=op)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi_, qi = p.bind(x=xi), q.bind(x=xi)
    s1, s2 = jno.np.sin(np.pi * xi), jno.np.sin(2 * np.pi * xi)
    f1 = (np.pi**2) * s1 + s2
    f2 = (4 * np.pi**2) * s2 + s1
    return d, jno.fem([ui.x * vi.x + pi_ * vi - f1 * vi, pi_.x * qi.x + ui * qi - f2 * qi, u(xb) - 0.0, p(xb) - 0.0])


@pytest.mark.parametrize("ou,op", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_coupled_1d_per_field_order_sizes_each_block(ou, op):
    """Each coupled field carries its OWN order, so the blocks are unequal — a P2 field adds one dof
    per element midpoint, a P1 field does not. The coupled 1D assembler used to be P1-only and refused
    ``order>1`` outright; sizing a P2 block with the vertex count would truncate it instead.

    ``fem.field_points`` reports each field's dof coordinates, because with mixed orders the block
    vector has no single coordinate list."""
    ms = 0.05
    d, fem = _coupled_orders(ms, ou, op)
    n_vert = int(np.asarray(d.mesh.points).shape[0])
    n_elem = int(np.asarray(d.mesh.cells_dict["line"]).shape[0])
    n_u = n_vert + (n_elem if ou == 2 else 0)
    n_p = n_vert + (n_elem if op == 2 else 0)
    assert fem.offsets == [0, n_u, n_u + n_p]
    assert fem.dofs == n_u + n_p

    fp = fem.field_points
    assert [int(np.asarray(x).shape[0]) for x in fp] == [n_u, n_p]

    # and it solves: each field recovered on its own dofs
    sol = np.asarray(fem.solve()).reshape(-1)
    xu, xp = np.asarray(fp[0]).reshape(-1), np.asarray(fp[1]).reshape(-1)
    assert np.max(np.abs(sol[:n_u] - np.sin(np.pi * xu))) < 1e-3
    assert np.max(np.abs(sol[n_u:] - np.sin(2 * np.pi * xp))) < 1e-3


@pytest.mark.parametrize("order,rate", [(1, 2.0), (2, 4.0)])
def test_coupled_1d_equal_order_converges_at_its_order(order, rate):
    """The point of P2 on a coupled system: the nodal error must fall at O(h^2k), not at the P1 rate.
    Measured on the u block, where both fields share the order so neither limits the other."""
    errs = []
    for ms in (0.1, 0.05, 0.025):
        fem = _coupled_orders(ms, order, order)[1]
        sol = np.asarray(fem.solve()).reshape(-1)
        n_u = fem.offsets[1]
        xu = np.asarray(fem.field_points[0]).reshape(-1)
        errs.append(float(np.max(np.abs(sol[:n_u] - np.sin(np.pi * xu)))))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(abs(r - rate) < 0.3 for r in rates), f"P{order} coupled rates {rates}, expected ~{rate}"


def test_coupled_1d_mixed_order_coupling_blocks_are_rectangular():
    """The structural consequence of mixed order: the off-diagonal coupling blocks are *rectangular*
    (n_u x n_p), which a shared-node-count layout could not represent. Both couplings must be present
    and non-zero, or the two equations are not actually coupled."""
    fem = _coupled_orders(0.1, 2, 1)[1]
    n_u = fem.offsets[1]
    A = _dense(fem.A)
    assert A.shape[0] == fem.dofs and A.shape[0] != 2 * n_u  # unequal blocks
    assert A[:n_u, n_u:].shape != A[n_u:, :n_u].shape  # rectangular, transposed shapes differ
    assert np.any(np.abs(A[:n_u, n_u:]) > 1e-12) and np.any(np.abs(A[n_u:, :n_u]) > 1e-12)


def test_coupled_1d_mixed_order_transient_seeds_every_dof():
    """A P2 field's initial state must be seeded at its element midpoints too, not only at the mesh
    vertices — otherwise the march starts from a state that is right on the vertices and zero between
    them. The IC is sampled on each field's own dof coordinates."""
    d = _line(0.05, time=(0.0, 0.02, 11))
    u, v = d.fem_symbols(names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi_, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x,
            pi_.t * qi + pi_.x * qi.x - u.bind(x=xi, t=ti) * qi,
            u(xb) - 0.0,
            p(xb) - 0.0,
            u(ci) - jno.np.sin(np.pi * ci),
            p(ci) - 0.0,
        ]
    )
    n_u = fem.offsets[1]
    xu = np.asarray(fem.field_points[0]).reshape(-1)
    assert len(xu) == n_u > int(np.asarray(d.mesh.points).shape[0]), "the P2 block is not midpoint-sized"
    s0 = np.asarray(fem.state0)
    assert s0.shape[0] == fem.dofs
    assert np.max(np.abs(s0[:n_u] - np.sin(np.pi * xu))) < 1e-12, "IC not seeded on every P2 dof"
    assert np.allclose(s0[n_u:], 0.0)
    traj = np.asarray(fem.solve().fn())
    assert traj.shape[1] == fem.dofs
    assert np.max(np.abs(traj[-1, :n_u])) < np.max(np.abs(traj[0, :n_u]))  # the heat block decays


# ==========================================================================
# coupled 1D — runtime parameters and neural coefficients
# ==========================================================================
def _coupled_parametric(kx, ms=0.01):
    """A coupled 1D system whose solution genuinely MOVES with the coefficient ``k``:

        -k u'' + p = pi^2 sin(pi x) + sin(2 pi x)
        -  p'' + u = 4 pi^2 sin(2 pi x) + sin(pi x),      u = p = 0 at both ends

    The sources are held fixed, so at ``k = 1`` the exact pair is ``(sin(pi x), sin(2 pi x))`` and any
    other ``k`` gives a different field. That is deliberate: the obvious manufactured coupled problem
    (``u = x``, ``p = 2x``) has ``u'' = 0``, so its solution is the same for *every* ``k`` — it cannot
    tell a live parameter from one frozen at its zero placeholder.
    """
    d = _line(ms)
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi_, qi = p.bind(x=xi), q.bind(x=xi)
    s1, s2 = jno.np.sin(np.pi * xi), jno.np.sin(2 * np.pi * xi)
    f1 = (np.pi**2) * s1 + s2
    f2 = (4 * np.pi**2) * s2 + s1
    return d, jno.fem(
        [
            kx(xi) * (ui.x * vi.x) + pi_ * vi - f1 * vi,
            pi_.x * qi.x + ui * qi - f2 * qi,
            u(xb) - 0.0,
            p(xb) - 0.0,
        ]
    )


def _const_k(c):
    return lambda _xi, _c=c: _c


def _eval_node(node):
    return np.asarray(jno.core([(node * 0.0).mae], domain=_dummy_domain()).eval([node])).reshape(-1)


def test_coupled_1d_reference_problem_actually_depends_on_k():
    """Guard on the guard: the coupled fixture must be k-sensitive, or every parametric assertion
    below would pass just as well against a coefficient baked in at its placeholder."""
    fem1 = _coupled_parametric(_const_k(1.0))[1]
    n = fem1.offsets[1]
    u1 = np.asarray(fem1.solve()).reshape(-1)
    x = np.asarray(fem1.points).reshape(-1)[:n]
    # at k = 1 it is the manufactured pair
    assert np.abs(u1[:n] - np.sin(np.pi * x)).max() < 1e-4
    assert np.abs(u1[n:] - np.sin(2 * np.pi * x)).max() < 1e-4
    u2 = np.asarray(_coupled_parametric(_const_k(2.5))[1].solve()).reshape(-1)
    assert np.abs(u1[:n] - u2[:n]).max() > 0.5, "the fixture is k-insensitive; it proves nothing"


@pytest.mark.parametrize("kval", [1.0, 2.5])
def test_coupled_1d_runtime_parameter_matches_the_constant_assembly(kval):
    """A ``jno.np.parameter`` coefficient on a COUPLED 1D system. The block element kernels publish the
    same ``runtime_parameter_tags``/``volume_vars`` keys the single-field ones do, so the shared
    evaluator reads them regardless of field layout — no block-specific machinery. Previously this
    raised: the coupled builder threaded no parameters at all.

    Checked at two *different* values, each against the constant-coefficient assembly at the same
    value: matching at one value alone is also what a frozen coefficient would do."""
    from jno.trace import FemLinearSystem

    k = jno.np.parameter((1,), name="k")
    k.initialize(jax.nn.initializers.constant(kval))
    fem = _coupled_parametric(lambda _xi, _k=k: _k)[1]
    assert isinstance(fem.operator, FemLinearSystem) and fem.operator.is_parametric
    assert list(fem.operator.runtime_parameter_exprs) == ["k"]

    node = fem.solve()
    assert not isinstance(node, jax.Array), "a parametric coupled 1D solve must be a trace node"
    ref = np.asarray(_coupled_parametric(_const_k(kval))[1].solve()).reshape(-1)
    assert np.max(np.abs(_eval_node(node) - ref)) < 1e-12


@pytest.mark.parametrize("cval", [1.0, 2.5])
def test_coupled_1d_neural_coefficient_matches_the_constant_assembly(cval):
    """A ``jno.nn.wrap`` coefficient on a coupled 1D system. A network never enters ``volume_vars`` (a
    weight pytree is cell-independent) — it rides the ``{name: module}`` table the kernel re-evaluates
    at the quad points, which the coupled ``local`` now carries too. A *constant* net must reproduce
    the scalar-coefficient block solve exactly, at more than one value."""
    net = _const_net_1d(cval)
    fem = _coupled_parametric(lambda xi, _n=net: _n(xi))[1]
    assert fem.operator.is_parametric
    ref = np.asarray(_coupled_parametric(_const_k(cval))[1].solve()).reshape(-1)
    assert np.max(np.abs(_eval_node(fem.solve()) - ref)) < 1e-12


def test_coupled_1d_inverse_recovers_a_scalar_parameter():
    """End to end: recover ``k`` in a COUPLED 1D system from full-field data through ``crux.solve``.
    ``∂u/∂k`` has to flow through the *block* solve for this to move at all."""
    import optax

    k_true, ms = 2.5, 0.02
    obs = jnp.asarray(np.asarray(_coupled_parametric(_const_k(k_true), ms=ms)[1].solve()).reshape(-1))

    k = jno.np.parameter((1,), name="k")
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(0.15))
    node = _coupled_parametric(lambda _xi, _k=k: _k, ms=ms)[1].solve()
    crux = jno.core([(node - obs).mae], domain=_dummy_domain())
    crux.solve(220)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - k_true) < 0.05, f"k not recovered through the coupled 1D inverse: {rec:.4f} vs {k_true}"


def test_coupled_1d_nonlinear_is_parametric_too():
    """A nonlinear coupled 1D form needs no extra machinery: the block residual already re-evaluates
    its coefficients from ``args``, and ``FemResidualOperator`` takes ``R(u, args)`` — so Newton runs
    on ``R(., k)`` and the implicit derivative gives ``du/dk``."""
    k = jno.np.parameter((1,), name="k")
    k.initialize(jax.nn.initializers.constant(1.0))
    d = _line(0.05)
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi_, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem(
        [
            k * (ui.x * vi.x) + ui * pi_ * vi - 2.0 * xi * xi * vi,
            pi_.x * qi.x + ui * ui * qi - xi * xi * qi,
            u(xb) - xb,
            p(xb) - 2.0 * xb,
        ]
    )
    assert not fem.is_linear
    assert list(fem.operator.runtime_parameter_exprs) == ["k"]


def test_coupled_1d_transient_parameter_fails_loud():
    """The coupled TRANSIENT block is assembled once, outside any per-args re-forming, so a parameter
    there would be read at its zero placeholder and silently baked in — refused, with the steady path
    named as the supported one. (Same rule the single-field transient applies to its mass.)"""
    k = jno.np.parameter((1,), name="k")
    k.initialize(jax.nn.initializers.constant(1.0))
    d = _line(0.1, time=(0.0, 0.02, 11))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi_, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    with pytest.raises(NotImplementedError, match="COUPLED 1D \\*transient\\*"):
        jno.fem(
            [
                ui.t * vi + k * (ui.x * vi.x),
                pi_.t * qi + pi_.x * qi.x - u.bind(x=xi, t=ti) * qi,
                u(xb) - 0.0,
                p(xb) - 0.0,
                u(ci) - 1.0,
                p(ci) - 0.0,
            ]
        )


# ==========================================================================
# coupled 1D — block layout published to the outside
# ==========================================================================
def test_coupled_1d_publishes_field_offsets():
    """``fem.offsets`` is the block layout every consumer slices a coupled solution by — the periodic
    reduction reduces block-wise through it, and post-processing splits the flat DOF vector with it.
    The 1D block assembler computed the layout internally but returned only ``(op, mode)``, so
    ``fem.offsets`` was ``None`` in 1D while 2D/3D reported ``[0, n, 2n]``. A coupled 1D system must be
    indistinguishable from a coupled 2D one here."""
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem([ui.x * vi.x + pi * vi - vi, pi.x * qi.x + ui * qi - qi, u(xb) - 0.0, p(xb) - 0.0])
    assert fem.offsets == [0, n, 2 * n], f"coupled 1D must publish its block offsets, got {fem.offsets}"
    assert fem.offsets[-1] == fem.dofs  # the layout accounts for every dof


def test_coupled_1d_transient_publishes_field_offsets():
    """Same contract on the transient block — the offsets come from the shared field layout, so they
    must survive the transient branch too (a separate return path)."""
    d = _line(0.1, time=(0.0, 0.02, 11))
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x,
            pi.t * qi + pi.x * qi.x - u.bind(x=xi, t=ti) * qi,
            u(xb) - 0.0,
            p(xb) - 0.0,
            u(ci) - jno.fn(lambda x: jnp.sin(jnp.pi * x), [ci]),
            p(ci) - 0.0,
        ]
    )
    assert fem.offsets == [0, n, 2 * n]


def test_coupled_1d_periodic_recovers_manufactured():
    """A periodic tie on a **coupled** 1D system. This is what the missing offsets blocked: the
    multi-field periodic reduction reduces each block as ``P_i^T A[i,j] P_j`` and reads the block
    bounds off ``fem.offsets``, so a ``None`` layout crashed with ``TypeError: 'NoneType' object is
    not iterable`` — a cryptic failure for a case 2D/3D handles.

    Manufactured, both fields 1-periodic on [0,1] with ``k = 2*pi``:
        -u'' +  u + p = (k^2+1) cos(kx) + sin(kx)      -> u* = cos(kx)
        -p'' + 2p + u = (k^2+2) sin(kx) + cos(kx)      -> p* = sin(kx)
    The positive reaction terms make the all-periodic system nonsingular (no constant null space)."""
    k = 2.0 * np.pi
    d = _line(0.005)
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    cos_kx, sin_kx = jno.np.cos(k * xi), jno.np.sin(k * xi)
    f1 = (k**2 + 1.0) * cos_kx + sin_kx
    f2 = (k**2 + 2.0) * sin_kx + cos_kx
    fem = jno.fem(
        [
            ui.x * vi.x + ui * vi + pi * vi - f1 * vi,
            pi.x * qi.x + 2.0 * pi * qi + ui * qi - f2 * qi,
            u(xl) - u(xr),
            p(xl) - p(xr),
        ]
    )
    assert fem._periodic is not None, "the coupled ties were not read as a periodic reduction"
    # one main/secondary identification per field -> the reduced space loses exactly 2 dofs
    assert fem._periodic["n_red"] == fem._periodic["n_full"] - 2

    sol = np.asarray(fem.solve()).reshape(-1)
    lo, mid, hi = fem.offsets
    assert len(sol) == hi, "the periodic solve must return the PROLONGED (full) block vector"
    xs = np.asarray(fem.points).reshape(-1)[lo:mid]
    u_ex, p_ex = np.cos(k * xs), np.sin(k * xs)
    ru = np.linalg.norm(sol[lo:mid] - u_ex) / np.linalg.norm(u_ex)
    rp = np.linalg.norm(sol[mid:hi] - p_ex) / np.linalg.norm(p_ex)
    assert ru < 5e-3, f"periodic coupled 1D u error {ru:.2e}"
    assert rp < 5e-3, f"periodic coupled 1D p error {rp:.2e}"
    # the tie is real: the identified endpoints carry the same value, per field
    assert abs(sol[lo] - sol[mid - 1]) < 1e-10
    assert abs(sol[mid] - sol[hi - 1]) < 1e-10


def test_coupled_1d_periodic_survives_publishing_the_assembly_records():
    """Regression guard. Once the coupled 1D path started publishing ``_fem_native_assembly_cells_all``
    (so a 1D transient could ride the adaptive remesher), the multi-field periodic reduction began
    taking the *facet* route with that connectivity — and ``_boundary_facets`` enumerated a simplex's
    facets as vertex PAIRS, which an interval's 2-column connectivity cannot index.

    A facet of a ``dim``-simplex has ``dim`` vertices, so in 1D it is a single endpoint. Pinned here
    because the two features are unrelated on their face: publishing an assembly record for the
    adaptivity path is not obviously a periodic-reduction concern."""
    from jno._fem import _boundary_facets

    pts = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    cells = np.column_stack([np.arange(4), np.arange(1, 5)])
    facets = _boundary_facets(pts, cells, 1, 1)
    assert facets is not None
    got = set(np.asarray(facets).reshape(-1).tolist())
    assert got == {0, 4}, f"an interval's boundary facets are its two endpoints, got {got}"


@pytest.mark.parametrize("space", ["Argyris", "Morley", "RT", "N1curl"])
def test_nonnodal_space_on_a_line_fails_loud(space):
    """The non-nodal push-forward assembler is built on triangles/tets, so asking for one of its
    families on a LINE mesh died with a bare ``KeyError: 'triangle'`` from the topology lookup — a
    cryptic failure for a reasonable request. It must name the dimension mismatch instead.

    Hermite is deliberately absent from this list: its 1D counterpart is the cubic beam element,
    which the 1D assembler builds (see the beam section below)."""
    d = _line(0.2)
    u, phi = d.fem_symbols(space=space)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    with pytest.raises(NotImplementedError, match="no 1D counterpart"):
        jno.fem([ui.x * vi.x - vi, u(xb) - 0.0])


# ==========================================================================
# the interval element spec and 1D facet machinery (the native-context pieces)
# ==========================================================================
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_interval_element_spec_is_a_valid_lagrange_basis(degree):
    """``lagrange_interval`` is the 1D sibling of ``lagrange_triangle``/``lagrange_tet``, built by the
    same basix builder — it is what lets the native ``fem_context`` (and hence VPINN) exist on a line.

    The invariants every Lagrange basis must satisfy: partition of unity, gradients summing to zero,
    second derivatives summing to zero, and a quadrature rule exact to the element's own degree."""
    from jno.utils.solver.fem_lagrange import lagrange_interval

    spec = lagrange_interval(degree)
    assert spec.n_dof == degree + 1
    N = np.asarray(spec.ref_values)[:, :, 0]
    dN = np.asarray(spec.ref_grads)[:, :, 0, 0]
    H = np.asarray(spec.ref_hess)[:, :, 0, 0, 0]
    qp = np.asarray(spec.quad_points).reshape(-1)
    qw = np.asarray(spec.quad_weights).reshape(-1)

    assert np.max(np.abs(N.sum(axis=1) - 1.0)) < 1e-12, "partition of unity"
    assert np.max(np.abs(dN.sum(axis=1))) < 1e-12, "gradients must sum to zero"
    assert np.max(np.abs(H.sum(axis=1))) < 1e-12, "second derivatives must sum to zero"
    assert abs(qw.sum() - 1.0) < 1e-12, "the reference interval has length 1"
    assert abs((qw * qp**degree).sum() - 1.0 / (degree + 1)) < 1e-12, f"rule not exact on xi^{degree}"


def test_interval_facets_are_the_two_endpoints_with_outward_normals():
    """A facet of a ``dim``-simplex has ``dim`` vertices, so an interval's facets are its two
    endpoints. A point has no tangent to rotate, so the unit candidate is ``+1`` and the *shared*
    away-from-the-apex flip picks the outward sign — which must come out ``-1`` at the left end and
    ``+1`` at the right."""
    from jno.utils.solver.fem_facets import build_facet_connectivity, compute_face_normals

    pts = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    cells = np.column_stack([np.arange(4), np.arange(1, 5)])
    conn = build_facet_connectivity(cells, "interval")
    assert conn.n_bfaces == 2
    ids = np.asarray(conn.face_nodes).reshape(-1)
    assert set(ids.tolist()) == {0, 4}, "the boundary of an interval mesh is its two ends"

    n = np.asarray(compute_face_normals(pts, conn, cells, "interval")).reshape(-1)
    by_id = dict(zip(ids.tolist(), n.tolist()))
    assert by_id[0] == pytest.approx(-1.0), "the left end's outward normal points -x"
    assert by_id[4] == pytest.approx(+1.0), "the right end's outward normal points +x"


def test_single_item_concat_returns_the_item():
    """``jnp_ops.concat`` skipped its fast path for a SINGLE operand and fell into a rank-alignment
    fallback written for two or more. That is unreachable in 2D/3D but not in 1D, where the canonical
    test-grad coefficient stacks one component per dimension — so a one-item stack — and the fallback
    re-entered trace-node construction inside the evaluator, hanging a 1D VPINN's loss.

    Checked directly on the op, because the symptom was a hang rather than a wrong number."""
    import jax.numpy as jnp

    from jno.jnp_ops import concat

    node = concat([jno.np.parameter((1,), name="_c1")])
    inner = node.fn if hasattr(node, "fn") else node.operation
    got = np.asarray(inner(jnp.asarray([[1.0], [2.0], [3.0]])))
    assert got.shape == (3, 1), "a one-item concat must return the item, trailing axis intact"
    assert np.allclose(got.reshape(-1), [1.0, 2.0, 3.0])


# ==========================================================================
# Hermite C1 cubic — the Euler-Bernoulli beam element
# ==========================================================================
def _beam(ms=0.1, q=1.0, EI=1.0, left="clamped", right="free"):
    """``EI w'''' = q`` on [0,1] as ``int EI w'' v'' - int q v``, on the C1 cubic Hermite space.

    ``left``/``right`` are the classical supports, which on this space are just *which of the node's
    two dofs* are pinned: ``clamped`` = deflection + slope, ``pinned`` = deflection only,
    ``guided`` = slope only, ``free`` = neither."""
    lap = jno.np.laplacian
    d = _line(ms)
    u, phi = d.fem_symbols(space="Hermite")
    xi = d.variable("interior", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    cons = [EI * lap(ui, [xi]) * lap(vi, [xi]) - q * vi]
    for end, kind in (("left", left), ("right", right)):
        xe = d.variable(end, split=True)[0]
        if kind in ("clamped", "pinned"):
            cons.append(u(xe) - 0.0)
        if kind in ("clamped", "guided"):
            cons.append(u.dn(xe) - 0.0)
    return d, jno.fem(cons)


def _beam_fields(d, fem):
    """``(x, w, theta)`` sorted by x — the deflection and slope dofs interleaved as ``2n`` / ``2n+1``."""
    sol = np.asarray(fem.solve()).reshape(-1)
    x = np.asarray(d.mesh.points)[:, 0]
    o = np.argsort(x)
    return x[o], sol[0::2][o], sol[1::2][o]


def test_hermite_beam_dof_layout():
    """The C1 cubic carries **two** dofs per vertex, ``(w, dw/dx)``, laid out node-major as
    ``2*node`` / ``2*node + 1``. Sharing the slope dof between neighbouring elements is what makes the
    space C1 — which is what gives the fourth-order operator a well-defined ``int w'' v''`` weak form.
    ``fem.points`` repeats each vertex so it lines up entry-for-entry with the solution vector."""
    d, fem = _beam(0.1)
    n_vert = int(np.asarray(d.mesh.points).shape[0])
    assert fem.dofs == 2 * n_vert
    pts = np.asarray(fem.points).reshape(-1)
    assert pts.shape[0] == 2 * n_vert
    assert np.allclose(pts[0::2], pts[1::2]), "both dofs of a node live at that node"
    assert np.allclose(np.sort(pts[0::2]), np.sort(np.asarray(d.mesh.points)[:, 0]))


def test_hermite_beam_cantilever_matches_analytic():
    """The headline case: a cantilever under uniform load. Clamped at x=0 (deflection AND slope pinned),
    free at x=1 (neither). The cubic Hermite beam is *nodally exact* for a uniform load, so both the tip
    deflection ``qL^4/8`` and the tip slope ``qL^3/6`` come out to machine precision — and the whole
    nodal deflection matches the analytic quartic ``q x^2 (6 - 4x + x^2) / 24``.

    The tip slope is the assertion a C0 space could not pass: it is a *solved dof*, not a
    post-processed difference."""
    d, fem = _beam(0.1, q=1.0, EI=1.0, left="clamped", right="free")
    x, w, th = _beam_fields(d, fem)
    assert abs(w[-1] - 1.0 / 8.0) < 1e-10, f"tip deflection {w[-1]:.8f} vs qL^4/8"
    assert abs(th[-1] - 1.0 / 6.0) < 1e-10, f"tip slope {th[-1]:.8f} vs qL^3/6"
    assert abs(w[0]) < 1e-12 and abs(th[0]) < 1e-12, "the clamped root must pin BOTH dofs"
    exact = (x**2) * (6.0 - 4.0 * x + x**2) / 24.0
    assert np.max(np.abs(w - exact)) < 1e-10


@pytest.mark.parametrize(
    "left,right,mid_exact",
    [("pinned", "pinned", 5.0 / 384.0), ("clamped", "clamped", 1.0 / 384.0)],
)
def test_hermite_beam_supports_change_the_answer(left, right, mid_exact):
    """The two other classical supports, which differ *only* in whether the slope dof is pinned:
    simply supported gives ``5qL^4/384`` at mid-span, clamped-clamped gives ``qL^4/384`` — a factor of
    five. Pinning the wrong dof (or silently pinning both) would land on the other value."""
    d, fem = _beam(0.1, left=left, right=right)
    x, w, _th = _beam_fields(d, fem)
    mid = int(np.argmin(np.abs(x - 0.5)))
    assert abs(w[mid] - mid_exact) < 1e-10, f"{left}/{right} mid-span {w[mid]:.8f} vs {mid_exact:.8f}"


def test_hermite_beam_free_end_is_genuinely_free():
    """A free end carries no essential condition at all, so both its dofs are solved. Contrast with the
    clamped end of the same beam: this is what pins that the slope condition is *optional* rather than
    silently applied everywhere."""
    d, fem = _beam(0.1, left="clamped", right="free")
    _x, w, th = _beam_fields(d, fem)
    assert abs(w[-1]) > 0.1 and abs(th[-1]) > 0.1, "the free end moved and rotated"


def test_hermite_beam_guided_end_pins_the_slope_only():
    """A guided (sliding) end pins the slope but not the deflection — the ``u.dn`` condition alone.
    Extreme on the BC axis: it must leave ``w`` free while forcing ``dw/dx = 0`` there."""
    d, fem = _beam(0.1, left="clamped", right="guided")
    _x, w, th = _beam_fields(d, fem)
    assert abs(th[-1]) < 1e-12, "the guided end must have zero slope"
    assert abs(w[-1]) > 1e-3, "the guided end must still deflect"


def test_hermite_beam_element_couples_four_dofs_and_stays_sparse():
    """Structure: a Hermite element couples its 4 dofs (both dofs of each endpoint), so the operator is
    banded and its nnz stays linear in the dof count — the element scatter must not have densified for
    the extra dof per node."""
    d, fem = _beam(0.005)
    A = fem.operator[0]
    n = A.shape[0]
    assert hasattr(A, "indices"), "the beam must assemble sparsely too"
    assert int(A.nse) < 20 * n, f"nnz={int(A.nse)} is not linear in n={n}"
    dense = _dense(A)
    # dof 2i couples only to 2i-3 .. 2i+3 (its own node's pair and the neighbours')
    rows, cols = np.nonzero(np.abs(dense) > 1e-12)
    assert np.max(np.abs(rows - cols)) <= 3, "the beam operator is not banded at bandwidth 3"


def test_hermite_beam_is_symmetric():
    """``int EI w'' v''`` is a symmetric bilinear form, and the Dirichlet elimination keeps it so —
    the same invariant the Lagrange path holds."""
    d, fem = _beam(0.1)
    A = _dense(fem.A)
    assert np.allclose(A, A.T, atol=1e-10)


# ==========================================================================
# complex 1D — the real-equivalent Re/Im split
# ==========================================================================
def _cx_line(ms=0.01):
    d = _line(ms)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    return d, u, phi, xi, d.variable("boundary", split=True)[0]


def test_complex_1d_helmholtz_recovers_manufactured():
    """A ``1j`` in a 1D weak form must route through the same real-equivalent split the 2D/3D and
    non-nodal paths use. Every other complex dispatch sits *below* the 1D branch, so before this a
    complex 1D form reached the real assembler: the stiffness came out ``complex128`` while the load
    scatter dropped its imaginary part (a numpy ``ComplexWarning``, no jNO error), and the solve then
    died inside jax's ``spsolve`` on a dtype mismatch.

    Manufactured, all-Neumann (no Dirichlet bookkeeping), ``u* = (1+0.5i) cos(pi x)``:
        c(-u'') + d u = f,  c = 1/(1 + 0.5i),  d = -(1 + 0.2i),  f = (pi^2 c + d) u*.
    Both the operator and the source are complex."""
    d, u, phi, xi, _ = _cx_line()
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    c = 1.0 / (1.0 + 1j * (0.5 + 0.0 * xi))  # traced -> stresses complex division through the trace
    d_coef = -(1.0 + 0.2j)
    amp = 1.0 + 0.5j
    g = jno.np.cos(np.pi * xi)
    f = (np.pi**2 * c + d_coef) * amp * g

    fem = jno.fem([c * (ui.x * vi.x) + d_coef * (ui * vi) - f * vi])
    assert fem.is_complex, "a 1j coefficient did not make the 1D system complex"

    u_num = np.asarray(fem.solve()).reshape(-1)
    assert np.iscomplexobj(u_num)
    pts = np.asarray(fem.points).reshape(-1)
    u_star = amp * np.cos(np.pi * pts)
    rel = float(np.linalg.norm(u_num - u_star) / np.linalg.norm(u_star))
    assert rel < 1e-3, f"complex 1D Helmholtz recovery rel-L2 {rel:.3e}"
    assert float(np.abs(u_num.imag).max()) > 0.1  # genuinely complex, not a real solve in disguise


def test_complex_1d_source_imaginary_part_is_not_dropped():
    """The precise silent path: a **real** operator with a **complex source**. Nothing about the
    matrix betrays the problem, so dropping ``Im(f)`` produced a plausible real field. ``-u'' = pi^2
    (1+2i) sin(pi x)``, ``u(0)=u(1)=0`` -> ``u = (1+2i) sin(pi x)``, which 1D P1 reproduces nodally."""
    d, u, phi, xi, xb = _cx_line(0.02)
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = (np.pi**2) * (1.0 + 2.0j) * jno.np.sin(np.pi * xi)
    fem = jno.fem([ui.x * vi.x - f * vi, u(xb) - 0.0])
    assert fem.is_complex

    u_num = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points).reshape(-1)
    exact = (1.0 + 2.0j) * np.sin(np.pi * pts)
    rel = float(np.linalg.norm(u_num - exact) / np.linalg.norm(exact))
    assert rel < 1e-6, f"complex 1D source rel-L2 {rel:.3e}"
    # the imaginary part is twice the real one -- a dropped Im(f) would leave it at zero
    assert float(np.abs(u_num.imag).max()) > 1.9


def test_complex_1d_real_dirichlet_pins_re_and_leaves_im_free():
    """A **real** essential value on a complex form is well-posed on the shared Dirichlet row set: the
    fused block imposes ``x_r - x_i = g`` and ``x_r + x_i = g``, i.e. ``Re u = g`` with ``Im u = 0``.
    ``-u'' + i u = 0``, ``u(0)=0``, ``u(1)=1``: the endpoints are pinned real, the interior is not."""
    d, u, phi, xi, _ = _cx_line(0.02)
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x + 1j * (ui * vi), u(xl) - 0.0, u(xr) - 1.0])
    assert fem.is_complex

    u_num = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points).reshape(-1)
    i0, i1 = int(np.argmin(pts)), int(np.argmax(pts))
    assert abs(u_num[i0] - 0.0) < 1e-12 and abs(u_num[i1] - 1.0) < 1e-12, "real Dirichlet not imposed"
    # the i*u reaction drives a genuinely complex interior -- otherwise this is a real solve
    assert float(np.abs(u_num.imag).max()) > 1e-3


def test_complex_1d_parametric_inverse_matches_the_constant_assembly():
    """The complex **inverse** in 1D: a ``jno.np.parameter`` inside the complex coefficient keeps both
    legs parametric, and the fused 2n block re-forms from the runtime args as a differentiable trace
    node. Evaluated at the parameter's value it must equal the constant-coefficient complex solve."""
    from jno.trace import FemLinearSystem

    def _build(sig):
        d, u, phi, xi, xb = _cx_line(0.02)
        ui, vi = u.bind(x=xi), phi.bind(x=xi)
        c = 1.0 / (1.0 + 1j * sig)
        return jno.fem([c * ui.x * vi.x - (np.pi**2) * jno.np.sin(np.pi * xi) * vi, u(xb) - 0.0])

    sig = jno.np.parameter((1,), name="sig")
    sig.initialize(jax.nn.initializers.constant(0.5))
    fem = _build(sig)
    assert fem.is_complex and isinstance(fem.operator, FemLinearSystem)

    node = fem.solve()
    assert not isinstance(node, jax.Array), "a parametric complex 1D solve must be a trace node"
    crux = jno.core([(node * 0.0).mae], domain=_dummy_domain())
    got = np.asarray(crux.eval([node])).reshape(-1)
    ref = np.asarray(_build(0.5).solve()).reshape(-1)
    assert np.iscomplexobj(got)
    assert np.max(np.abs(got - ref)) < 1e-12, "parametric complex 1D disagrees with the constant one"


def test_complex_1d_scope_limits_fail_loud():
    """Every path the Re/Im split does NOT cover raises rather than dropping the imaginary part —
    the same rule the non-nodal complex branch follows."""
    # transient
    d = _line(0.1, time=(0.0, 0.02, 11))
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    with pytest.raises(NotImplementedError, match="complex \\*transient\\* 1D"):
        jno.fem([ui.t * vi + 1j * (ui.x * vi.x), u(xb) - 0.0, u(ci) - 1.0])

    # nonlinear
    d2, u2, phi2, xi2, xb2 = _cx_line(0.1)
    ui2, vi2 = u2.bind(x=xi2), phi2.bind(x=xi2)
    with pytest.raises(NotImplementedError, match="complex \\*nonlinear\\* 1D"):
        jno.fem([ui2.x * vi2.x + 1j * (ui2 * ui2 * vi2) - vi2, u2(xb2) - 0.0])

    # coupled
    d3 = _line(0.1)
    a, b = d3.fem_symbols(names=("a", "b"))
    p, q = d3.fem_symbols(names=("p", "q"))
    xi3 = d3.variable("interior", split=True)[0]
    xb3 = d3.variable("boundary", split=True)[0]
    ai, bi = a.bind(x=xi3), b.bind(x=xi3)
    pi_, qi = p.bind(x=xi3), q.bind(x=xi3)
    with pytest.raises(NotImplementedError, match="complex COUPLED 1D"):
        jno.fem([ai.x * bi.x + 1j * pi_ * bi - bi, pi_.x * qi.x - qi, a(xb3) - 0.0, p(xb3) - 0.0])

    # complex essential value: inexpressible on the shared Dirichlet rows, so refused rather than
    # silently imposed as Re(g) (which is what the 2D/3D float cast does).
    d4, u4, phi4, xi4, _ = _cx_line(0.1)
    xr4 = d4.variable("right", split=True)[0]
    xl4 = d4.variable("left", split=True)[0]
    ui4, vi4 = u4.bind(x=xi4), phi4.bind(x=xi4)
    with pytest.raises(NotImplementedError, match="COMPLEX essential value"):
        jno.fem([ui4.x * vi4.x + 0.0j * (ui4 * vi4), u4(xl4) - 0.0, u4(xr4) - (1.0 + 2.0j)])


# ==========================================================================
# tagged boundaries, and the assembled tangent
# ==========================================================================
def test_a_coordinate_tag_becomes_a_dirichlet_boundary_region():
    """``d.tag(...)`` must register a BOUNDARY region in 1-D as it does in 2-D/3-D.

    A 1-D boundary facet **is** a vertex, but the registration searched only ``edges`` (2-D) /
    ``triangles`` (3-D) and so returned without registering anything. The failure was silent in the
    worst way: the tag still existed as a sampling region, ``d.variable(name)`` still returned its
    coordinate, and nothing complained until ``jno.fem`` rejected ``u(tag) - g`` as a *whole-domain*
    residual — an error that blames the residual for a defect in the tag.
    """
    d = _line(0.1)
    d.tag("right_end", lambda x: x > 1.0 - 1e-9)
    assert "right_end" in d._boundary_regions, "the tag never became a boundary region"
    # and it must carry the outward normal, like every other boundary tag: +1 at the right end
    assert float(np.asarray(d.normals_by_tag["right_end"]).ravel()[0]) == pytest.approx(1.0)

    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]  # built-in
    xr = d.variable("right_end", split=True)[0]  # ours
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, u(xl) - 0.0, u(xr) - 1.0])
    sol = np.asarray(fem.solve()).reshape(-1)
    assert np.allclose(sol, _x(d), atol=1e-10), "-u'' = 0 with u(0)=0, u(1)=1 is u = x"


def test_a_tag_matching_no_boundary_node_stays_interior():
    """The early return is still right when the predicate genuinely misses the boundary — a tag over
    the middle of the line is an interior sampling region, not a Dirichlet one."""
    d = _line(0.1)
    d.tag("middle", lambda x: jnp.abs(x - 0.5) < 1e-9)
    assert "middle" not in d._boundary_regions


def _allen_cahn(eps=0.15, T=2.0, nstep=24, mesh_size=0.02, wide=0.30):
    """Transient Allen-Cahn, u_t = eps^2 u_xx + u - u^3, from a deliberately over-wide interface.

    The stationary profile is tanh((x - 1/2) / (sqrt2 eps)) — Allen & Cahn, Acta Metall. 27 (1979)
    1085-1095, Sec. 2. Nonlinear AND transient, so it exercises the transient assembled tangent; the
    right edge comes from ``d.tag`` rather than the built-in tag so one test covers both 1-D gaps at
    once, which is how the two were actually met."""
    profile = lambda x: np.tanh((x - 0.5) / (np.sqrt(2.0) * eps))  # noqa: E731
    d = _line(mesh_size, time=(0.0, T, nstep))
    d.tag("right_end", lambda x: x > 1.0 - 1e-9)
    u, v = d.fem_symbols()
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[-1]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right_end", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi)
    fem = jno.fem(
        [
            ui.t * vi + eps**2 * (ui.x * vi.x) + (u**3 - u) * vi,
            u(xl) - profile(0.0),
            u(xr) - profile(1.0),
            u(ci) - jno.np.tanh((ci - 0.5) / (np.sqrt(2.0) * wide)),
        ]
    )
    return fem, profile(np.asarray(fem.points).reshape(-1))


def test_the_assembled_tangent_of_a_nonlinear_1d_problem_is_sparse():
    """``newton(direct=True)`` factorises the ASSEMBLED tangent, and its contract is a **BCOO**.

    1-D handed it a dense global ``jacfwd`` instead. Inside the Newton ``lax.while_loop`` that made
    ``sparse_lu_solve`` call ``BCOO.fromdense`` on a tracer, whose ``nse`` cannot be concrete, so every
    ``direct=True`` solve died with a ``ConcretizationTypeError`` — and it is an ``O(N^2)`` tangent
    besides, on the one dimension where node counts are meant to be large.
    """
    fem, ref = _allen_cahn()
    traj = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True)).fn())
    assert np.isfinite(traj).all(), "the direct Newton produced non-finite values"
    rel = lambda s: float(np.linalg.norm(ref - s) / np.linalg.norm(ref))  # noqa: E731
    assert rel(traj[0]) > 0.2, "the initial interface should start visibly too wide"
    assert rel(traj[-1]) < 0.02, f"did not sharpen onto the stationary profile: rel_L2 = {rel(traj[-1]):.3e}"


def test_a_steady_nonlinear_1d_problem_takes_the_direct_newton():
    """Same assembled-tangent contract on the steady nonlinear path (a different jacobian site).

    ``-u'' + u^3 = 0`` with ``u(0)=0, u(1)=1``: monotone between the two ends, which the linear
    solution is not once the cubic bites."""
    d = _line(0.05)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x + (ui**3) * vi, u(xl) - 0.0, u(xr) - 1.0])
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True))).reshape(-1)
    x = _x(d)
    assert np.isfinite(sol).all()
    assert sol[np.argmin(x)] == pytest.approx(0.0, abs=1e-10)
    assert sol[np.argmax(x)] == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.diff(sol[np.argsort(x)]) > -1e-12), "the solution must stay monotone"
    # the cubic sink pulls the profile BELOW the straight line -u''=0 would give
    assert np.max(np.sort(x) - sol[np.argsort(x)]) > 1e-3


# ==========================================================================
# tag resolution
# ==========================================================================
@pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
def test_a_tagged_endpoint_carries_its_dirichlet_at_either_precision(x64):
    """A ``domain.tag`` predicate must select the same node whether or not x64 is on.

    It did not. The 1D assembler read the mesh points into a **JAX** array to evaluate the tag's
    location function, which is float32 with x64 off — and ``1 - 1e-9`` is not representable there,
    it rounds to exactly ``1.0``, so the strict ``>`` below matched no node. The essential condition
    was then dropped in **silence**: ``fem.classification`` still reported ``dirichlet@right_edge``,
    the solve succeeded, and the answer was a different BVP (``-u'' = 0`` with one end free is
    constant, so ``u(1)`` came back at the *left* value). Every other test in this file opts into
    x64 through the autouse fixture, which is precisely why it went unnoticed — hence the explicit
    parametrization here.

    ``-u'' = 0`` with ``u(0) = -1``, ``u(1) = +1`` is nodally exact, so a missing condition cannot
    hide inside discretisation error.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64)
    try:
        d = _line(0.02)
        d.tag("right_edge", lambda x: x > 1.0 - 1e-9)
        u, phi = d.fem_symbols()
        xi = d.variable("interior", split=True)[0]
        xl = d.variable("left", split=True)[0]
        xr = d.variable("right_edge", split=True)[0]
        ui, vi = u.bind(x=xi), phi.bind(x=xi)
        fem = jno.fem([ui.x * vi.x, u(xl) - (-1.0), u(xr) - 1.0])
        sol = np.asarray(fem.solve()).reshape(-1)
        x = np.asarray(fem.points)[:, 0]
        assert sol[np.argmax(x)] == pytest.approx(1.0, abs=1e-4), "the tagged end lost its Dirichlet value"
        assert sol[np.argmin(x)] == pytest.approx(-1.0, abs=1e-4)
        assert np.linalg.norm(sol - (2 * x - 1)) / np.linalg.norm(2 * x - 1) < 1e-3
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_an_essential_condition_matching_no_node_is_refused():
    """An essential condition that selects nothing is a different problem, not a no-op.

    The tag below is deliberately off the domain (the line ends at x = 1) — what a mis-specified
    predicate looks like, now that the precision bug above can no longer produce one. Two guards
    catch it and either is fine, so this pins the *refusal*, not the wording: a tag matching nothing
    never registers as a boundary region, so ``jno.fem`` rejects the term while classifying it; and
    if one ever did register while still resolving to no node — exactly the shape of the float32 bug
    — ``_essential_node_ids`` raises at DOF resolution. What must never happen is a solve that
    succeeds with the condition quietly missing.
    """
    d = _line(0.05)
    d.tag("nowhere", lambda x: x > 5.0)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xn = d.variable("nowhere", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    with pytest.raises(ValueError, match="matched no mesh node|must live on a boundary region"):
        jno.fem([ui.x * vi.x, u(xl) - 0.0, u(xn) - 1.0]).solve()


def test_tag_node_mask_resolves_a_fine_tolerance_without_x64():
    """The unit behind both tests above: the predicate is evaluated in float64, not at the ambient dtype."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        d = _line(0.02)
        d.tag("right_edge", lambda x: x > 1.0 - 1e-9)
        pts = np.asarray(d.mesh.points)
        mask = d.tag_node_mask("right_edge", pts)
        assert mask.sum() == 1, "a 1e-9 tolerance at x = 1 is below float32 resolution"
        assert pts[mask, 0] == pytest.approx(1.0)
        # the built-in tag names the same node, and always did (it never evaluates a user tolerance)
        assert np.array_equal(mask, d.tag_node_mask("right", pts))
        assert d.tag_node_mask("no_such_tag", pts) is None
    finally:
        jax.config.update("jax_enable_x64", prev)
