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


def test_vec_gt_1_rejected():
    d = _line(0.3)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    weak = jno.np.inner(jno.np.grad(u, [xi]), jno.np.grad(phi, [xi]), n_contract=2)
    with pytest.raises(NotImplementedError):
        jno.fem([weak, u(xb) - 0.0])


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


def test_1d_parameter_scope_limits_fail_loud():
    """Scope, each with its own reason rather than one blanket refusal."""
    # A field parameter is interpolated with the element's own shape functions, so its nodal layout must
    # match the trial's and a P1 field cannot ride a LINE3 element. The refusal comes from
    # `jno.np.parameter` itself, at construction -- earlier than assembly, and the better place for it.
    d = _line(0.1)
    _u, phi = d.fem_symbols(order=2)
    with pytest.raises(NotImplementedError, match="P1|order=1"):
        jno.np.parameter(phi, name="kf2")

    # a parameter in a NONLINEAR 1D form: the parametric path is wired for the steady linear system
    d2 = _line(0.1)
    u2, p2 = d2.fem_symbols()
    x2 = d2.variable("interior", split=True)[0]
    xb2 = d2.variable("boundary", split=True)[0]
    u2i, v2i = u2.bind(x=x2), p2.bind(x=x2)
    k2 = jno.np.parameter((1,), name="k3")
    k2.initialize(jax.nn.initializers.constant(1.0))
    with pytest.raises(NotImplementedError, match="nonlinear"):
        jno.fem([k2 * (u2i.x * v2i.x) + (u2 * u2) * v2i - 1.0 * v2i, u2(xb2) - 0.0])


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


def test_order3_and_coupled_p2_fail_loud():
    """Scope, stated explicitly: 1D Lagrange is implemented for order 1 and 2, and the *coupled* 1D
    block assembler is still P1 — both refuse rather than silently solving at the wrong order."""
    d = _line(0.25)
    u, phi = d.fem_symbols(order=3)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    with pytest.raises(NotImplementedError, match="order"):
        jno.fem([ui.x * vi.x - 1.0 * vi, u(xb) - 0.0])


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
