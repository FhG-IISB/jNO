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
    # one master/slave identification per field -> the reduced space loses exactly 2 dofs
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


@pytest.mark.parametrize("space", ["Hermite", "Argyris", "Morley", "RT", "N1curl"])
def test_nonnodal_space_on_a_line_fails_loud(space):
    """The non-nodal push-forward assembler is built on triangles/tets, so asking for one of its
    families on a LINE mesh died with a bare ``KeyError: 'triangle'`` from the topology lookup — a
    cryptic failure for a reasonable request. It must name the dimension mismatch instead."""
    d = _line(0.2)
    u, phi = d.fem_symbols(space=space)
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    with pytest.raises(NotImplementedError, match="no 1D counterpart"):
        jno.fem([ui.x * vi.x - vi, u(xb) - 0.0])


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
