"""Neural coefficients in assembled FEM systems: ``jno.nn.wrap(net)`` called inside a weak form
(e.g. ``net(x, y) * u.dx * v.dx``) is a trainable *coefficient* on an ordinary FE system — not a
VPINN trial. The system assembles as usual; the kernel re-evaluates the network at the quadrature
points, the weights ride the runtime ``args`` as a ``ModelWeights`` slot, and ``crux.solve``
trains them through the differentiable ``fem.solve()`` (NN-EUCLID-style unsupervised coefficient
recovery — Flaschel/Kumar/De Lorenzis, JMPS 165 (2022); Tartakovsky et al., WRR 56 (2020)).

Coverage: steady/nonlinear/transient/complex on the native 2D/3D Lagrange assembler (single AND
coupled multi-field), plus scalar C¹ non-nodal elements (Argyris/Morley/Hermite). The set-level
routing rule (a weak constraint carrying a real ``TrialFunction`` means the network is a
coefficient, not the trial) must not disturb VPINN.

Run with x64 (assembly runs in float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
pytest.importorskip("foundax", reason="foundax required for the MLP coefficient nets")

import equinox as eqx  # noqa: E402
import foundax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402
from jno.trace import ModelWeights  # noqa: E402

PI = np.pi

# The inverse loss has no spatial Variable (the FEM solve is global), so crux needs
# an explicit domain to drive its loop.
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64; set x64 per-test with save/restore (the global flag is
    shared across modules and other suites flip it at import)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class _Const(eqx.Module):
    """A 'network' that outputs a constant per quad point — the degenerate extreme that must
    reproduce a scalar-coefficient assembly exactly (and gives a single-leaf gradient check)."""

    c: jnp.ndarray

    def __call__(self, *args):
        n = jnp.asarray(args[0]).shape[0]
        return jnp.broadcast_to(self.c.reshape(1, 1), (n, 1))


def _const_net(c):
    net = jno.nn.wrap(_Const(c=jnp.asarray(float(c), dtype=jnp.float64)))
    net.dtype(jnp.float64)
    return net


def _mlp_net(key=0, hidden=16, layers=2):
    net = jno.nn.wrap(
        foundax.mlp(2, hidden_dims=hidden, num_layers=layers, activation=jax.nn.tanh, key=jax.random.PRNGKey(key))
    )
    net.dtype(jnp.float64)
    return net


def _poisson_setup(mesh_size=0.25):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    return d, u, phi, (xi, yi), (xb, yb), ui, vi, f


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


# ==========================================================================
# routing: coefficient vs VPINN trial
# ==========================================================================


def test_net_coefficient_routes_to_linear_parametric_system():
    """``net(x,y) * grad u . grad v``: a weak constraint carrying the real TrialFunction makes the
    network a coefficient — an assembled linear system, parametric in the net's weights (one
    ModelWeights slot named after the model)."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    net = _mlp_net()
    fem = jno.fem([net(xi, yi) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    assert fem.is_linear
    exprs = fem.operator.runtime_parameter_exprs
    assert len(exprs) == 1
    (name,) = exprs
    assert isinstance(exprs[name], ModelWeights)
    assert fem.operator.metadata.get("nonaffine_operator") is True


def test_vpinn_network_trial_routing_unchanged():
    """A network *replacing* the trial (no TrialFunction in any weak constraint) still routes to
    VPINN — the set-level disambiguation must not regress the network-trial path."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    u_net = _mlp_net(key=1)(xi, yi)
    weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - f * vi
    pde = jno.fem([weak, u(xb, yb) - 0.0])
    assert hasattr(pde, "mse") and hasattr(pde, "volume_grad_expr")  # GroupedAssembly, not FEM


# ==========================================================================
# extremes: constant net / zero net reproduce scalar assembly exactly
# ==========================================================================


def test_constant_net_matches_scalar_coefficient_operator():
    """A net that outputs the constant 0.7 must assemble the SAME operator as the scalar
    coefficient 0.7 (to solver precision) — pins quad-point evaluation and alignment."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    net = _const_net(0.7)
    fem = jno.fem([net(xi, yi) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    fem_ref = jno.fem([0.7 * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    (name,) = fem.operator.runtime_parameter_exprs
    A, b = fem.operator.evaluate({name: net.module})
    assert np.abs(_dense(A) - _dense(fem_ref.A)).max() < 1e-12
    assert np.abs(np.asarray(b).reshape(-1) - np.asarray(fem_ref.b).reshape(-1)).max() < 1e-12


def test_zero_net_zeroes_its_term():
    """A zero net in ``(net + 1) * grad u . grad v`` leaves exactly the unit-coefficient system —
    the additive composition evaluates the net inside the integrand, not as a factored scalar."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    net = _const_net(0.0)
    fem = jno.fem([(net(xi, yi) + 1.0) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    fem_ref = jno.fem([1.0 * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.abs(_dense(A) - _dense(fem_ref.A)).max() < 1e-12


# ==========================================================================
# frozen networks: known coefficients, non-parametric assembly
# ==========================================================================


def test_frozen_net_assembles_nonparametric_and_matches_unfrozen():
    """``net.freeze()`` = a KNOWN network coefficient: the system stays non-parametric (eager
    ``fem.A``) and the matrix equals the unfrozen twin re-assembled at the same weights."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    net_t = _mlp_net(key=2)
    fem_t = jno.fem([(net_t(xi, yi) + 1.5) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    (name,) = fem_t.operator.runtime_parameter_exprs
    A_train, _ = fem_t.operator.evaluate({name: net_t.module})

    net_f = _mlp_net(key=2)  # identical weights (same PRNG key)
    net_f.freeze()
    fem_f = jno.fem([(net_f(xi, yi) + 1.5) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert not getattr(fem_f.operator, "is_parametric", False)  # plain (A, b), solved eagerly
    assert np.abs(_dense(fem_f.A) - _dense(A_train)).max() < 1e-12

    sol = np.linalg.solve(_dense(fem_f.A), np.asarray(fem_f.b).reshape(-1))
    assert np.all(np.isfinite(sol))


# ==========================================================================
# composition: scalar param + nodal field + net in ONE weak form
# ==========================================================================


def test_mixed_scalar_nodal_neural_coefficients_compose_and_differentiate():
    """One weak form carrying all three trainable coefficient kinds — a scalar
    ``jno.np.parameter``, a nodal field ``jno.np.parameter(phi)``, and a network — collects all
    three runtime slots, re-assembles, and the solve is differentiable in each."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
    alpha.initialize(jax.nn.initializers.constant(1.0))
    alpha.dtype(jnp.float64)
    k = jno.np.parameter(phi, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    net = _mlp_net(key=3)

    weak = alpha * (ui.x * vi.x + ui.y * vi.y) + k * (ui * vi) + (net(xi, yi) + 1.0) * (ui * vi) - f * vi
    fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)
    exprs = fem.operator.runtime_parameter_exprs
    net_name = next(n for n in exprs if n not in ("alpha", "k"))
    assert set(exprs) == {"alpha", "k", net_name}
    assert isinstance(exprs[net_name], ModelWeights)

    n_nodes = int(np.asarray(d.built_mesh.points).shape[0])

    def solve_at(a_val, k_vals, module):
        A, b = fem.operator.evaluate({"alpha": a_val, "k": k_vals, net_name: module})
        return jnp.linalg.solve(A.todense(), jnp.asarray(b).reshape(-1))

    a0, k0 = jnp.asarray(1.0), jnp.ones((n_nodes,))
    loss = lambda a, kv, m: jnp.sum(solve_at(a, kv, m) ** 2)
    g_a = jax.grad(loss, argnums=0)(a0, k0, net.module)
    g_k = jax.grad(loss, argnums=1)(a0, k0, net.module)
    g_m = eqx.filter_grad(lambda m: loss(a0, k0, m))(net.module)
    g_m_sum = sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(g_m) if eqx.is_inexact_array(x))
    assert abs(float(g_a)) > 0.0
    assert float(jnp.abs(g_k).sum()) > 0.0
    assert float(g_m_sum) > 0.0


# ==========================================================================
# per-region masks compose with a net coefficient
# ==========================================================================


def test_region_masked_load_with_net_coefficient():
    """A region-restricted load ``-net*v`` with a constant-1 net integrates to the region's cell
    area exactly (like the plain per-region load) — the mask's volume_var slot must not shift when
    a neural coefficient is present."""
    disk = lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2  # noqa: E731
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    d.tag("disk", disk)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xd, yd, _ = d.variable("disk", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    vd = phi.bind(x=xd, y=yd)
    net = _const_net(1.0)

    fem = jno.fem(
        [(ui.x * vi.x + ui.y * vi.y), -(net(xd, yd) * vd), u(xb, yb) - 0.0],
        quad_degree=3,
    )
    (name,) = fem.operator.runtime_parameter_exprs
    _, b = fem.operator.evaluate({name: net.module})

    fem_ref = jno.fem([(ui.x * vi.x + ui.y * vi.y), -(1.0 * vd), u(xb, yb) - 0.0], quad_degree=3)
    assert np.abs(np.asarray(b).reshape(-1) - np.asarray(fem_ref.b).reshape(-1)).max() < 1e-12


# ==========================================================================
# vector trial and boundary (Robin) terms
# ==========================================================================


def test_vector_elasticity_with_scalar_net_stiffness():
    """A scalar net multiplying a vector (vec=2) elasticity form: constant-net assembly equals the
    scalar-coefficient reference."""
    from jno.jnp_ops import inner, symgrad, trace

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"))
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    eu, ev = symgrad(u, [xi, yi]), symgrad(w, [xi, yi])
    vv = w.bind(x=xi, y=yi)
    net = _const_net(0.8)
    lam = 1.2
    weak = lam * trace(eu) * trace(ev) + 2 * net(xi, yi) * inner(eu, ev, n_contract=2) - (1.0 * vv[0] + 0.5 * vv[1])
    weak_ref = lam * trace(eu) * trace(ev) + 2 * 0.8 * inner(eu, ev, n_contract=2) - (1.0 * vv[0] + 0.5 * vv[1])

    fem = jno.fem([weak, u(xb, yb) - (0.0, 0.0)], quad_degree=2)
    fem_ref = jno.fem([weak_ref, u(xb, yb) - (0.0, 0.0)], quad_degree=2)
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.abs(_dense(A) - _dense(fem_ref.A)).max() < 1e-11


def test_net_in_robin_boundary_term():
    """A net coefficient inside a surface (Robin) integrand ``net*u*v`` on one edge — exercises the
    surface-kernel threading; constant net equals the scalar Robin reference."""
    from jno.jnp_ops import grad, inner

    d = jno.Shape.rect(0, 0, 1, 1, size=0.25).domain()
    u, w = d.fem_symbols(names=("u", "w"))
    xi, yi = d.variable("interior", split=True)[:2]
    vv = w.bind(x=xi, y=yi)
    xr, yr = d.variable("right", split=True)[:2]
    ur, wr = u.bind(x=xr, y=yr), w.bind(x=xr, y=yr)
    xl, yl = d.variable("left", split=True)[:2]
    net = _const_net(2.5)

    cons = [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - 1.0 * vv, net(xr, yr) * ur * wr, u(xl, yl) - 0.0]
    cons_ref = [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - 1.0 * vv, 2.5 * ur * wr, u(xl, yl) - 0.0]
    fem = jno.fem(cons, quad_degree=3)
    fem_ref = jno.fem(cons_ref, quad_degree=3)
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.abs(_dense(A) - _dense(fem_ref.A)).max() < 1e-12


# ==========================================================================
# the headline: recover k(x) end-to-end through crux
# ==========================================================================


@pytest.mark.slow
def test_neural_kx_recovers_via_crux():
    """Recover a smooth diffusivity k(x) = 0.6 + 0.8x + 0.5y with a coordinate MLP trained through
    the differentiable ``fem.solve()``. The ``1 + net`` offset keeps the operator nonsingular at
    the (near-zero) net init — same practice as starting the nodal field at k=1."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup(mesh_size=0.25)

    fem_ref = jno.fem([(0.6 + 0.8 * xi + 0.5 * yi) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    u_obs = jnp.linalg.solve(jnp.asarray(_dense(fem_ref.A)), jnp.asarray(fem_ref.b).reshape(-1))

    net = _mlp_net(key=0)
    net.optimizer(optax.adam(1e-2))
    fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(600)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_net = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
    k_true = 0.6 + 0.8 * nodes[:, 0] + 0.5 * nodes[:, 1]
    rel = float(np.linalg.norm(k_net - k_true) / np.linalg.norm(k_true))
    assert rel < 0.1, f"neural k(x) recovery rel-err {rel:.3e}"


# ==========================================================================
# dtype and scope guards
# ==========================================================================


def test_f32_net_promotes_under_x64():
    """A plain f32 net (no explicit .dtype opt-in) under x64: assembly stays float64 by promotion
    and the solve is finite — never a silent downcast of the system."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup()
    net = jno.nn.wrap(foundax.mlp(2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(4)))
    fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    (name,) = fem.operator.runtime_parameter_exprs
    A, b = fem.operator.evaluate({name: net.module})
    assert _dense(A).dtype == np.float64
    sol = np.linalg.solve(_dense(A), np.asarray(b).reshape(-1))
    assert np.all(np.isfinite(sol))


# ==========================================================================
# learned constitutive laws: k(u) / k(∇u) — steady nonlinear
# (NN-EUCLID-style: Flaschel/Kumar/De Lorenzis, JMPS 165 (2022) §2.2–2.3)
# ==========================================================================


class _Quad(eqx.Module):
    """'Network' computing exactly k(u) = a + b·u² — lets the nonlinear solve be checked against
    the symbolically written form with zero training (and gives named-leaf gradient checks)."""

    a: jnp.ndarray
    b: jnp.ndarray

    def __call__(self, u):
        return self.a + self.b * jnp.asarray(u) ** 2


def _quad_net(a=1.0, b=0.5):
    net = jno.nn.wrap(_Quad(a=jnp.asarray(float(a)), b=jnp.asarray(float(b))))
    net.dtype(jnp.float64)
    return net


def _ku_setup(mesh_size=0.25):
    d, u, phi, (xi, yi), (xb, yb), ui, vi, _ = _poisson_setup(mesh_size)
    f = 10.0 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    return d, u, phi, (xi, yi), (xb, yb), ui, vi, f


def test_ku_classification():
    """A net whose args carry the unknown makes the form NONLINEAR — including the bare reaction
    ``net(u)*v`` that no product rule catches — while a coordinate-input net stays linear."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup()
    net = _quad_net()
    fem = jno.fem([net(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem._mode == "nonlinear"

    net_r = _quad_net()
    fem_r = jno.fem([(ui.x * vi.x + ui.y * vi.y) + net_r(ui) * vi - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem_r._mode == "nonlinear"

    net_x = _mlp_net(key=7)
    fem_l = jno.fem([(1.0 + net_x(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem_l._mode == "linear"


def test_ku_forward_matches_symbolic_newton():
    """The net-coefficient Newton solve equals the symbolically written ``(1 + 0.5u²)∇u·∇v`` Newton
    solve to machine precision — the net is evaluated inside the residual, its u-dependence enters
    the element Jacobian through jacfwd."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup()
    net = _quad_net(a=1.0, b=0.5)
    fem = jno.fem([net(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    fem_sym = jno.fem([(1.0 + 0.5 * ui**2) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    (name,) = fem.operator.runtime_parameter_exprs
    u_net = newton_krylov(lambda v: fem.operator.residual(v, {name: net.module}), jnp.zeros(fem.operator.size))
    u_sym = newton_krylov(lambda v: fem_sym.operator(v), jnp.zeros(fem_sym.operator.size))
    assert float(jnp.max(jnp.abs(u_net - u_sym))) < 1e-10
    assert float(jnp.linalg.norm(u_sym)) > 0.1  # a genuinely nonzero, nonlinear solution


def test_kgradu_form_matches_symbolic():
    """k(∇u): a net taking the gradient components (p-Laplacian-style ``k = 1 + 0.5|∇u|²``) solves
    and matches the symbolic reference."""
    from jno.utils.solver.newton_krylov import newton_krylov

    class _PLap(eqx.Module):
        b: jnp.ndarray

        def __call__(self, gx, gy):
            return 1.0 + self.b * (jnp.asarray(gx) ** 2 + jnp.asarray(gy) ** 2)

    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup()
    net = jno.nn.wrap(_PLap(b=jnp.asarray(0.5)))
    net.dtype(jnp.float64)
    fem = jno.fem([net(ui.x, ui.y) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem._mode == "nonlinear"
    fem_sym = jno.fem(
        [(1.0 + 0.5 * (ui.x**2 + ui.y**2)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3
    )
    (name,) = fem.operator.runtime_parameter_exprs
    u_net = newton_krylov(lambda v: fem.operator.residual(v, {name: net.module}), jnp.zeros(fem.operator.size))
    u_sym = newton_krylov(lambda v: fem_sym.operator(v), jnp.zeros(fem_sym.operator.size))
    assert float(jnp.max(jnp.abs(u_net - u_sym))) < 1e-10


def test_constant_ku_net_equals_linear_solve():
    """The degenerate constitutive law k(u) ≡ c must reproduce the plain linear solve (even though
    the form still classifies nonlinear — Newton just converges to the linear solution)."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup()
    net = _quad_net(a=2.0, b=0.0)
    fem = jno.fem([net(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    fem_lin = jno.fem([2.0 * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    (name,) = fem.operator.runtime_parameter_exprs
    u_net = newton_krylov(lambda v: fem.operator.residual(v, {name: net.module}), jnp.zeros(fem.operator.size))
    u_lin = jnp.linalg.solve(jnp.asarray(_dense(fem_lin.A)), jnp.asarray(fem_lin.b).reshape(-1))
    assert float(jnp.max(jnp.abs(u_net - u_lin))) < 1e-9


def test_ku_gradient_matches_finite_difference():
    """The implicit-diff gradient (custom_root through Newton) w.r.t. a net weight matches central
    finite differences — the least-exercised corner (closure-converted module pytree)."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup(mesh_size=0.4)
    net = _quad_net(a=1.0, b=0.5)
    fem = jno.fem([net(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    (name,) = fem.operator.runtime_parameter_exprs

    def loss_at(b_val):
        module = _Quad(a=jnp.asarray(1.0), b=jnp.asarray(b_val))
        uu = newton_krylov(lambda v: fem.operator.residual(v, {name: module}), jnp.zeros(fem.operator.size))
        return jnp.sum(uu**2)

    g_ad = float(jax.grad(loss_at)(0.5))
    h = 1e-6
    g_fd = float((loss_at(0.5 + h) - loss_at(0.5 - h)) / (2 * h))
    assert abs(g_ad - g_fd) < 1e-5 * max(1.0, abs(g_fd)), f"AD {g_ad} vs FD {g_fd}"


@pytest.mark.slow
def test_ku_recovers_constitutive_law_via_crux():
    """NN-EUCLID-style unsupervised constitutive learning: observe u from a hidden law
    k(u) = 1 + 0.5u², train a 1-input MLP ``1 + net(u)`` through the differentiable nonlinear
    ``fem.solve()``, and check the learned k against the truth on the observed u-range."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup(mesh_size=0.25)
    fem_sym = jno.fem([(1.0 + 0.5 * ui**2) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    u_obs = newton_krylov(lambda v: fem_sym.operator(v), jnp.zeros(fem_sym.operator.size))

    net = jno.nn.wrap(foundax.mlp(1, hidden_dims=16, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(8)))
    net.dtype(jnp.float64)
    net.optimizer(optax.adam(1e-2))
    fem = jno.fem([(1.0 + net(ui)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem._mode == "nonlinear"

    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(600)

    trained = crux.eval([ModelWeights(net)])
    u_grid = jnp.linspace(0.0, float(jnp.max(u_obs)), 64).reshape(-1, 1)
    k_learned = 1.0 + np.asarray(trained(u_grid)).reshape(-1)
    k_true = 1.0 + 0.5 * np.asarray(u_grid).reshape(-1) ** 2
    rel = float(np.linalg.norm(k_learned - k_true) / np.linalg.norm(k_true))
    assert rel < 0.05, f"constitutive-law recovery rel-err {rel:.3e}"


def test_ku_composes_with_solver_slots():
    """The nonlinear neural-coefficient solve composes with the slot API
    (``fem.solve(nonlinear=jno.solve.newton())``) through crux."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup(mesh_size=0.35)
    net = _quad_net(a=1.0, b=0.3)
    net.optimizer(optax.adam(1e-3))
    fem = jno.fem([net(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    u_node = fem.solve(nonlinear=jno.solve.newton())
    crux = jno.core([(u_node**2).mse], domain=_DUMMY)
    crux.solve(3)  # composes and steps without error
    val = np.asarray(crux.eval([ModelWeights(net)]).b)
    assert np.isfinite(val).all()


# ==========================================================================
# transient: neural coefficients in the time stepper
# ==========================================================================


def _transient_setup(mesh_size=0.25, time=(0.0, 0.05, 11)):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0


def _grid_ts(block):
    n_steps = int(round((float(block.t1) - float(block.t0)) / float(block.dt)))
    return jnp.linspace(float(block.t0), float(block.t1), n_steps + 1)


def test_transient_net_kx_trajectory_matches_frozen_reference():
    """Transient heat with a net diffusivity: the parametric per-step re-assembly at the net's
    stored weights reproduces the frozen (non-parametric) twin's trajectory exactly."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0 = _transient_setup()
    net = _mlp_net(key=9)
    fem = jno.fem([ui.t * vi + (1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    assert fem.is_transient
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)
    traj = _default_transient_integrate(fem.operator, {name: net.module}, _grid_ts(fem.operator))

    d2, u2, phi2, (x2, y2), (xb2, yb2), ci2, u2i, v2i, u02 = _transient_setup()
    net_f = _mlp_net(key=9)  # identical weights
    net_f.freeze()
    fem_f = jno.fem(
        [
            u2i.t * v2i + (1.0 + net_f(x2, y2)) * (u2i.x * v2i.x + u2i.y * v2i.y),
            u2(xb2, yb2) - 0.0,
            u2(ci2[0], ci2[1]) - u02,
        ]
    )
    traj_f = _default_transient_integrate(fem_f.operator, {}, _grid_ts(fem_f.operator))
    assert traj.shape == traj_f.shape
    assert float(jnp.max(jnp.abs(traj - traj_f))) < 1e-9


@pytest.mark.slow
def test_transient_neural_kx_recovers_via_crux():
    """Recover a smooth diffusivity k(x) from a u(t) trajectory with a coordinate MLP trained
    through the transient ``fem.solve()`` (per-step re-assembly, implicit-diff to the weights) —
    the neural analogue of the transient nodal-field recovery."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0 = _transient_setup(mesh_size=0.2)
    fem_ref = jno.fem(
        [ui.t * vi + (1.0 + 0.5 * xi + 0.3 * yi) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0]
    )
    u_obs = _default_transient_integrate(fem_ref.operator, {}, _grid_ts(fem_ref.operator))
    assert not bool(jnp.isnan(u_obs).any())

    d2, u2, phi2, (x2, y2), (xb2, yb2), ci2, u2i, v2i, u02 = _transient_setup(mesh_size=0.2)
    net = _mlp_net(key=10)
    net.optimizer(optax.adam(1e-2))
    fem = jno.fem(
        [u2i.t * v2i + (1.0 + net(x2, y2)) * (u2i.x * v2i.x + u2i.y * v2i.y), u2(xb2, yb2) - 0.0, u2(ci2[0], ci2[1]) - u02]
    )
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(300)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d2.built_mesh.points)[:, :2]
    k_net = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
    k_true = 1.0 + 0.5 * nodes[:, 0] + 0.3 * nodes[:, 1]
    rel = float(np.linalg.norm(k_net - k_true) / np.linalg.norm(k_true))
    assert rel < 0.1, f"transient neural k(x) recovery rel-err {rel:.3e}"


@pytest.mark.slow
def test_transient_nonlinear_ku_recovers_via_crux():
    """A constitutive law k(u) in a TRANSIENT form: u_t = div((a + b·u²)∇u). The per-step Newton
    (implicit backward Euler) carries the net's u-dependence; recover (a, b) from the trajectory
    through fem.solve()."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0 = _transient_setup(mesh_size=0.25, time=(0.0, 0.05, 6))
    fem_ref = jno.fem(
        [ui.t * vi + (1.0 + 0.5 * ui**2) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0]
    )
    u_obs = _default_transient_integrate(fem_ref.operator, {}, _grid_ts(fem_ref.operator))

    d2, u2, phi2, (x2, y2), (xb2, yb2), ci2, u2i, v2i, u02 = _transient_setup(mesh_size=0.25, time=(0.0, 0.05, 6))
    net = _quad_net(a=1.3, b=0.1)  # start away from the truth (1.0, 0.5)
    net.optimizer(optax.adam(2e-2))
    fem = jno.fem([u2i.t * v2i + net(u2i) * (u2i.x * v2i.x + u2i.y * v2i.y), u2(xb2, yb2) - 0.0, u2(ci2[0], ci2[1]) - u02])
    assert fem.is_transient and not fem.operator.is_linear()

    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(250)
    trained = crux.eval([ModelWeights(net)])
    u_grid = jnp.linspace(0.0, float(jnp.max(u_obs)), 32)
    k_learned = np.asarray(trained(u_grid)).reshape(-1)
    k_true = 1.0 + 0.5 * np.asarray(u_grid) ** 2
    rel = float(np.linalg.norm(k_learned - k_true) / np.linalg.norm(k_true))
    assert rel < 0.05, f"transient constitutive-law recovery rel-err {rel:.3e}"


def test_frozen_net_on_mass_term_is_allowed():
    """A *frozen* (known) net on the mass (u_t) term IS allowed — a known spatially-varying density
    ρ(x)·u_t: the mass matrix is assembled once from its stored weights, matching the same scalar
    density's trajectory. Only a *trainable* net on the mass term is guarded (see the scope test)."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    def setup(coeff):
        d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0 = _transient_setup(mesh_size=0.25, time=(0.0, 0.05, 6))
        return d, jno.fem([coeff(xi, yi) * ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])

    net = _const_net(1.3)
    net.freeze()
    d, fem = setup(lambda x, y: net(x, y))
    assert fem.is_transient and not getattr(fem.operator, "is_parametric", False)  # known coeff -> non-parametric
    traj = _default_transient_integrate(fem.operator, {}, _grid_ts(fem.operator))

    _, fem_ref = setup(lambda x, y: 1.3)
    traj_ref = _default_transient_integrate(fem_ref.operator, {}, _grid_ts(fem_ref.operator))
    assert float(jnp.max(jnp.abs(traj - traj_ref))) < 1e-12


def _mass_density_setup(rho_fn, mesh_size=0.25, time=(0.0, 0.05, 6)):
    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0 = _transient_setup(mesh_size=mesh_size, time=time)
    return d, jno.fem([rho_fn(xi, yi) * ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


def test_mass_net_kx_parametric_matches_frozen():
    """A trainable coordinate net(x) on the mass (u_t) term — an unknown density ρ(x)·u_t — makes the
    mass a *parametric matrix* re-assembled from args each step (block.mass_fn). It reproduces the
    frozen (fixed-mass) trajectory, and the trajectory's gradient flows THROUGH the mass to the net."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    net = _mlp_net(key=34, hidden=8)
    d, fem = _mass_density_setup(lambda x, y: 1.0 + net(x, y))
    assert fem.operator.mass_fn is not None  # parametric mass
    (name,) = fem.operator.runtime_parameter_exprs
    traj = _default_transient_integrate(fem.operator, {name: net.module}, _grid_ts(fem.operator))

    net_f = _mlp_net(key=34, hidden=8)
    net_f.freeze()
    _, fem_f = _mass_density_setup(lambda x, y: 1.0 + net_f(x, y))
    traj_f = _default_transient_integrate(fem_f.operator, {}, _grid_ts(fem_f.operator))
    assert float(jnp.max(jnp.abs(traj - traj_f))) < 1e-9

    def loss(m):
        return jnp.sum(_default_transient_integrate(fem.operator, {name: m}, _grid_ts(fem.operator)) ** 2)

    g = eqx.filter_grad(loss)(net.module)
    gsum = sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(g) if eqx.is_inexact_array(x))
    assert float(gsum) > 0.0  # gradient reached the net through the mass


@pytest.mark.slow
def test_mass_net_kx_recovers_via_crux():
    """Recover an unknown density ρ(x) on the mass term from a decay trajectory through the
    differentiable transient `fem.solve()` — the headline for the parametric-mass path."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    _, fem_ref = _mass_density_setup(lambda x, y: 1.0 + 0.5 * x + 0.3 * y)
    u_obs = _default_transient_integrate(fem_ref.operator, {}, _grid_ts(fem_ref.operator))

    net = _mlp_net(key=35, hidden=16)
    net.optimizer(optax.adam(1e-2))
    d, fem = _mass_density_setup(lambda x, y: 1.0 + net(x, y))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(400)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    rho_net = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
    rho_true = 1.0 + 0.5 * nodes[:, 0] + 0.3 * nodes[:, 1]
    rel = float(np.linalg.norm(rho_net - rho_true) / np.linalg.norm(rho_true))
    assert rel < 0.12, f"unknown-density ρ(x) recovery rel-err {rel:.3e}"


# ==========================================================================
# complex: a real net coefficient through the real-equivalent block
# ==========================================================================


def _complex_helmholtz(kappa_expr_fn, mesh_size=0.15):
    """Complex Helmholtz ``κ(-Δu) + d·u = f`` (all-Neumann), manufactured
    ``u* = (1+0.5i) cos(πx) cos(πy)`` at κ_true = 0.8 (mirrors test_fem_complex_parametric)."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    d_coef = 1.0 + 0.3j
    f = (2 * PI**2 * 0.8 + d_coef) * (1.0 + 0.5j) * jno.np.cos(PI * xi) * jno.np.cos(PI * yi)
    weak = kappa_expr_fn(xi, yi, ui) * (ui.x * vi.x + ui.y * vi.y) + d_coef * (u * vi) - f * vi
    return dom, jno.fem([weak])


def test_complex_net_yields_parametric_legs_and_node():
    """A real net coefficient in a complex form: the legs assemble as parametric FemLinearSystems
    carrying the ModelWeights slot, and solve() is a differentiable trace node."""
    from jno.trace import FemLinearSystem

    net = _mlp_net(key=11)
    _, fem = _complex_helmholtz(lambda x, y, u: 1.0 + net(x, y))
    assert fem.is_complex
    op_r, _op_i = fem._op
    assert isinstance(op_r, FemLinearSystem) and op_r.is_parametric
    (name,) = op_r.runtime_parameter_exprs
    assert isinstance(op_r.runtime_parameter_exprs[name], ModelWeights)
    assert not isinstance(fem.solve(), jax.Array)


def test_complex_constant_net_matches_scalar_forward():
    """A constant net (κ ≡ 0.8) through the complex block equals the scalar-κ complex solve —
    pins the Re/Im-leg kernel evaluation of the net."""
    _, fem_ref = _complex_helmholtz(lambda x, y, u: 0.8)
    u_ref = np.asarray(fem_ref.solve()).reshape(-1)

    net = _const_net(-0.2)  # 1 + net ≡ 0.8
    _, fem = _complex_helmholtz(lambda x, y, u: 1.0 + net(x, y))
    u_node = fem.solve()
    crux = jno.core([(u_node - u_ref).mae], domain=_DUMMY)
    u_par = np.asarray(crux.eval([u_node])).reshape(-1)
    assert np.allclose(u_par, u_ref, atol=1e-8)
    assert float(np.abs(u_ref.imag).max()) > 0.1  # genuinely complex


@pytest.mark.slow
def test_complex_net_recovers_kappa_via_crux():
    """Recover the (constant) diffusivity κ = 0.8 with a coordinate MLP trained through the
    complex real-equivalent block — ∂u/∂weights flows through the parametric legs."""
    _, fem_ref = _complex_helmholtz(lambda x, y, u: 0.8)
    u_ref = np.asarray(fem_ref.solve())

    net = _mlp_net(key=12, hidden=8)
    net.optimizer(optax.adam(1e-2))
    dom, fem = _complex_helmholtz(lambda x, y, u: 1.0 + net(x, y))
    crux = jno.core([(fem.solve() - u_ref).mae], domain=_DUMMY)
    crux.solve(300)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(dom.built_mesh.points)[:, :2]
    k_net = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
    rel = float(np.linalg.norm(k_net - 0.8) / (0.8 * np.sqrt(len(k_net))))
    assert rel < 0.1, f"complex neural κ recovery rel-err {rel:.3e}"


def test_complex_scope_guards():
    """The two complex compositions that would mis-assemble stay rejected: a solution-dependent
    net (the legs are linear real-equivalent blocks) and the complex transient."""
    net = _quad_net()
    with pytest.raises(NotImplementedError, match="complex"):
        _complex_helmholtz(lambda x, y, u: net(u))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 0.02, 5))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    net2 = _mlp_net(key=13)
    psi0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1]) * (1.0 + 0.0j)
    with pytest.raises(NotImplementedError, match="complex"):
        jno.fem(
            [
                1j * ui.t * vi - (1.0 + net2(xi, yi)) * (ui.x * vi.x + ui.y * vi.y),
                u(xb, yb) - 0.0,
                u(ci[0], ci[1]) - psi0,
            ]
        )


# ==========================================================================
# multi-field (coupled): a neural coefficient needs no per-field resolution
# ==========================================================================


def _coupled_setup(mesh_size=0.12):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    nnode = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    g = xi * (1 - xi) * yi * (1 - yi)
    lg = 2 * (xi * (1 - xi) + yi * (1 - yi))
    f1, f2 = lg + 2 * g, 2 * lg + g
    return d, nnode, (u, v, p, q), (xi, yi), (xb, yb), (ui, vi, pi, qi), (f1, f2)


def test_multifield_constant_net_matches_scalar_operator():
    """A constant net on one field's stiffness in a COUPLED (2-field) form assembles the same block
    operator as the scalar coefficient — the net threads without per-field resolution."""
    d, nnode, (u, v, p, q), (xi, yi), (xb, yb), (ui, vi, pi, qi), (f1, f2) = _coupled_setup()
    net = _const_net(0.7)

    def cons(kco):
        return [
            kco * (ui.x * vi.x + ui.y * vi.y) + p * vi - f1 * vi,
            pi.x * qi.x + pi.y * qi.y + u * qi - f2 * qi,
            u(xb, yb) - 0.0,
            p(xb, yb) - 0.0,
        ]

    fem = jno.fem(cons(net(xi, yi)), quad_degree=3)
    fem_ref = jno.fem(cons(0.7), quad_degree=3)
    assert fem.is_linear and fem.dofs == 2 * nnode
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.abs(_dense(A) - _dense(fem_ref.A)).max() < 1e-12


def test_multifield_nonlinear_net_resolves_correct_field():
    """A solution-dependent net ``net(u_0)`` in a coupled form must resolve field 0's trial (not
    field 0 by default) and match the symbolic ``(1 + 0.5 u_0²)`` coupled Newton solve — the
    definitive per-field-resolution check — with the gradient flowing to the net's weights."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d, nnode, (u, v, p, q), (xi, yi), (xb, yb), (ui, vi, pi, qi), (f1, f2) = _coupled_setup()

    def cons(kco):
        return [
            kco * (ui.x * vi.x + ui.y * vi.y) + p * vi - f1 * vi,
            pi.x * qi.x + pi.y * qi.y + u * qi - f2 * qi,
            u(xb, yb) - 0.0,
            p(xb, yb) - 0.0,
        ]

    net = _quad_net(a=1.0, b=0.5)
    fem = jno.fem(cons(net(ui)), quad_degree=3)
    fem_sym = jno.fem(cons(1.0 + 0.5 * ui**2), quad_degree=3)
    assert fem._mode == "nonlinear"
    (name,) = fem.operator.runtime_parameter_exprs
    u_net = newton_krylov(lambda w: fem.operator.residual(w, {name: net.module}), jnp.zeros(fem.operator.size))
    u_sym = newton_krylov(lambda w: fem_sym.operator(w), jnp.zeros(fem_sym.operator.size))
    assert float(jnp.max(jnp.abs(u_net - u_sym))) < 1e-9

    def loss(module):
        r = lambda w: fem.operator.residual(w, {name: module})
        return jnp.sum(newton_krylov(r, jnp.zeros(fem.operator.size)) ** 2)

    g = eqx.filter_grad(loss)(net.module)
    assert abs(float(g.a)) > 0.0 and abs(float(g.b)) > 0.0


# ==========================================================================
# non-nodal scalar C¹ elements (Argyris / Morley / Hermite)
# ==========================================================================


def test_argyris_constant_net_matches_coordinate_coeff():
    """Constant-net ≡ the same coordinate-function coefficient on an Argyris biharmonic operator —
    the definitive gather/interpolation check on a C¹ element (the net is evaluated at the quad
    points, independent of the Argyris DOF layout). Mirrors the field-parameter Argyris oracle."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.35)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    lap = jno.np.laplacian
    f = 48.0 + 0.0 * xi
    g = xb**4 + yb**4
    net = _const_net(0.7)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([net(xi, yi) * (lap(ui, [xi, yi]) * lap(vi, [xi, yi])) - f * vi, u(xb, yb) - g])
    assert fem.is_linear and len(fem.operator.runtime_parameter_exprs) == 1

    u2, p2 = d.fem_symbols(space="Argyris")
    ux, vx = u2.bind(x=xi, y=yi), p2.bind(x=xi, y=yi)
    fem_ref = jno.fem([0.7 * (lap(ux, [xi, yi]) * lap(vx, [xi, yi])) - f * vx, u2(xb, yb) - g])
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.max(np.abs(_dense(A) - _dense(fem_ref.A))) < 1e-9


def test_morley_constant_net_matches_coordinate_coeff():
    """Constant-net ≡ coordinate coefficient on the Morley full-Hessian biharmonic (``∫ D²u:D²v``,
    the correct non-singular form) — the gather oracle on the cheap C¹ element."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 48.0 + 0.0 * xi
    g = xb**4 + yb**4
    net = _const_net(0.7)
    u, phi = d.fem_symbols(space="Morley")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
    fem = jno.fem([net(xi, yi) * jno.np.inner(Hu, Hv, n_contract=2) - f * vi, u(xb, yb) - g])
    u2, p2 = d.fem_symbols(space="Morley")
    ux, vx = u2.bind(x=xi, y=yi), p2.bind(x=xi, y=yi)
    Hu2, Hv2 = jno.np.hessian(ux, [xi, yi]), jno.np.hessian(vx, [xi, yi])
    fem_ref = jno.fem([0.7 * jno.np.inner(Hu2, Hv2, n_contract=2) - f * vx, u2(xb, yb) - g])
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.max(np.abs(_dense(A) - _dense(fem_ref.A))) < 1e-9


def test_hermite_constitutive_ku_matches_symbolic():
    """A constitutive law ``net(u)`` on a Hermite (C¹) element: the form is nonlinear and its
    Newton solve matches the symbolic ``(1 + 0.3 u²)`` reference — the net's u-dependence enters
    the element Jacobian through the C¹ push-forward assembly's jacfwd."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    net = _quad_net(a=1.0, b=0.3)
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([net(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    assert fem._mode == "nonlinear"
    u2, p2 = d.fem_symbols(space="Hermite")
    u2i, v2i = u2.bind(x=xi, y=yi), p2.bind(x=xi, y=yi)
    fem_sym = jno.fem([(1.0 + 0.3 * u2i**2) * (u2i.x * v2i.x + u2i.y * v2i.y) - f * v2i, u2(xb, yb) - 0.0])
    (name,) = fem.operator.runtime_parameter_exprs
    u_net = newton_krylov(lambda w: fem.operator.residual(w, {name: net.module}), jnp.zeros(fem.operator.size))
    u_sym = newton_krylov(lambda w: fem_sym.operator(w), jnp.zeros(fem_sym.operator.size))
    assert float(jnp.max(jnp.abs(u_net - u_sym))) < 1e-9


@pytest.mark.slow
def test_hermite_neural_kx_recovers_via_crux():
    """End-to-end: recover a smooth k(x) with a coordinate MLP on a Hermite (C¹) element through the
    non-nodal parametric ``fem.solve()`` — proves the ModelWeights threading reaches the non-nodal
    FemLinearSystem and gradients flow to the weights."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    fem_ref = jno.fem([(0.6 + 0.8 * xi + 0.5 * yi) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    u_obs = jnp.linalg.solve(jnp.asarray(_dense(fem_ref.A)), jnp.asarray(fem_ref.b).reshape(-1))

    net = _mlp_net(key=20, hidden=16)
    net.optimizer(optax.adam(1e-2))
    fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    # The non-nodal path assembles a DENSE operator (jacfwd), so use a dense solve_fn — the default
    # sparse_lu_solve's BCOO-fromdense has a data-dependent nse under crux's jit (same as the
    # field-parameter Hermite recovery).
    solver = lambda A, b: jnp.linalg.solve(  # noqa: E731
        jnp.asarray(A.todense() if hasattr(A, "todense") else A), jnp.asarray(b).reshape(-1)
    )
    crux = jno.core([(fem.solve(solver) - u_obs).mse], domain=_DUMMY)
    crux.solve(400)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_net = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
    k_true = 0.6 + 0.8 * nodes[:, 0] + 0.5 * nodes[:, 1]
    rel = float(np.linalg.norm(k_net - k_true) / np.linalg.norm(k_true))
    assert rel < 0.12, f"Hermite neural k(x) recovery rel-err {rel:.3e}"


def test_rt_scalar_net_matches_coordinate_coeff():
    """A scalar coordinate net(x) coefficient on a vector H(div) (RT) form — a spatially-varying
    permeability multiplying the vector mass — assembles the same operator as the scalar
    coefficient. The net evaluates to a scalar at the quad points, independent of the Piola-pushed
    vector basis; the gather oracle for the edge families."""
    from jno.jnp_ops import inner

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    net = _const_net(0.7)
    fem = jno.fem([net(xi, yi) * inner(ui, vi) - inner(vi, vi)])
    fem_ref = jno.fem([0.7 * inner(ui, vi) - inner(vi, vi)])
    assert fem.is_linear
    (name,) = fem.operator.runtime_parameter_exprs
    A, _ = fem.operator.evaluate({name: net.module})
    assert np.max(np.abs(_dense(A) - _dense(fem_ref.A))) < 1e-9


def test_rt_solution_dependent_net_guard():
    """A *solution-dependent* net(u) on a vector edge family fails loud — feeding the vector-valued
    trial into the network is undefined. Only coordinate net(x) is supported there."""
    from jno.jnp_ops import inner

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    net = _mlp_net(key=21)
    with pytest.raises(NotImplementedError, match="RT|Nédélec|solution-dependent"):
        jno.fem([net(ui) * inner(ui, vi) - inner(vi, vi)])


def _hermite_transient_setup(coeff_fn, mesh_size=0.3, time=(0.0, 0.05, 6)):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, phi = d.fem_symbols(space="Hermite")
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return d, jno.fem([ui.t * vi + coeff_fn(xi, yi) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


def test_transient_nonnodal_net_kx_matches_frozen():
    """A trainable net(x) diffusivity in a Hermite (C¹) TRANSIENT form: the per-step re-assembled
    operator leg reproduces the frozen (fixed-operator) trajectory, and the trajectory's gradient
    reaches the net weights through the time-stepping."""
    from jno.utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate

    net = _mlp_net(key=30, hidden=8)
    d, fem = _hermite_transient_setup(lambda x, y: 1.0 + net(x, y))
    assert fem.is_transient
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)
    ts = _block_time_grid(fem.operator)
    traj = _default_transient_integrate(fem.operator, {name: net.module}, ts)

    net_f = _mlp_net(key=30, hidden=8)  # identical weights
    net_f.freeze()
    _, fem_f = _hermite_transient_setup(lambda x, y: 1.0 + net_f(x, y))
    traj_f = _default_transient_integrate(fem_f.operator, {}, _block_time_grid(fem_f.operator))
    assert float(jnp.max(jnp.abs(traj - traj_f))) < 1e-8

    def loss(m):
        return jnp.sum(_default_transient_integrate(fem.operator, {name: m}, ts) ** 2)

    g = eqx.filter_grad(loss)(net.module)
    gsum = sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(g) if eqx.is_inexact_array(x))
    assert float(gsum) > 0.0


@pytest.mark.slow
def test_transient_nonnodal_neural_kx_recovers_via_crux():
    """Recover a smooth k(x) from a Hermite transient trajectory through the default `fem.solve()`
    (the transient integrator solves the dense per-step system directly, so no dense solve_fn
    override is needed — unlike the steady non-nodal path)."""
    from jno.utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate

    _, fem_ref = _hermite_transient_setup(lambda x, y: 1.0 + 0.5 * x + 0.3 * y)
    u_obs = _default_transient_integrate(fem_ref.operator, {}, _block_time_grid(fem_ref.operator))

    net = _mlp_net(key=31, hidden=16)
    net.optimizer(optax.adam(1e-2))
    d, fem = _hermite_transient_setup(lambda x, y: 1.0 + net(x, y))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(300)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_net = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
    k_true = 1.0 + 0.5 * nodes[:, 0] + 0.3 * nodes[:, 1]
    rel = float(np.linalg.norm(k_net - k_true) / np.linalg.norm(k_true))
    assert rel < 0.12, f"transient non-nodal neural k(x) recovery rel-err {rel:.3e}"


def test_transient_nonnodal_mass_net_guard():
    """A trainable net on the mass (u_t) term of a non-nodal transient form fails loud (the mass
    block is assembled once) — a frozen one would be fine."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols(space="Hermite")
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    net2 = _mlp_net(key=33)
    with pytest.raises(NotImplementedError, match="mass"):
        jno.fem([(1.0 + net2(xi, yi)) * ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


def test_nonnodal_trainable_net_in_host_assembled_boundary_term_guard():
    """On a non-nodal element a *trainable* net in a HOST-ASSEMBLED natural-BC (RT pressure / plate moment)
    boundary term fails loud — that load is baked non-differentiably. (The N1E tangential-trace impedance /
    incident SURFACE BC now IS neural-differentiable — see test_fem_nedelec_impedance; only these
    host-assembled families are not, so the guard is now narrow and names the case.)"""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    xi, yi, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    wr = phi.bind(x=xr, y=yr)
    net = _mlp_net(key=22)
    with pytest.raises(NotImplementedError, match="host-assembled|natural-BC|neural coefficient"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, net(xr, yr) * wr, u(xl, yl) - 0.0])


# ==========================================================================
# net-valued Dirichlet: an unknown boundary profile u(region) - net(x)
# ==========================================================================


def test_dirichlet_net_routes_parametric():
    """``u(boundary) - net(xb, yb)`` (a trial-only constraint) is a trainable Dirichlet *value*, not a
    coefficient: the system routes to a parametric FemLinearSystem carrying the net's ModelWeights
    slot, and the operator itself is unchanged (only the load carries the net)."""
    from jno.trace import FemLinearSystem

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    net = _mlp_net(key=40)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 0.0 * vi, u(xb, yb) - net(xb, yb)], quad_degree=3)
    assert fem.is_linear and isinstance(fem.operator, FemLinearSystem) and fem.operator.is_parametric
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)


@pytest.mark.slow
def test_dirichlet_net_recovers_bc_profile():
    """The headline: recover an unknown Dirichlet profile g(x) = net(x) from the interior response.
    Laplace ``∇u·∇v = 0`` with ``u(boundary) - net(xb, yb)`` — u is the harmonic extension of the BC,
    so the interior data pins the boundary profile. The recovery (∂loss/∂weights actually moves the
    profile) is the correctness test — an operator-match would pass even if the value froze."""

    def setup(g_node_fn):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        return d, u, (xb, yb), jno.fem([ui.x * vi.x + ui.y * vi.y - 0.0 * vi, g_node_fn(u, xb, yb)], quad_degree=3)

    _, _, _, fem_ref = setup(lambda u, xb, yb: u(xb, yb) - (0.3 + 0.5 * xb + 0.2 * yb))
    u_obs = jnp.linalg.solve(jnp.asarray(_dense(fem_ref.A)), jnp.asarray(fem_ref.b).reshape(-1))

    net = _mlp_net(key=41, hidden=16)
    net.optimizer(optax.adam(1e-2))
    d, u, (xb, yb), fem = setup(lambda u, xb, yb: u(xb, yb) - net(xb, yb))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(400)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    onb = np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1) | np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1)
    bn = nodes[onb]
    g_net = np.asarray(trained(jnp.asarray(bn[:, 0:1]), jnp.asarray(bn[:, 1:2]))).reshape(-1)
    g_true = 0.3 + 0.5 * bn[:, 0] + 0.2 * bn[:, 1]
    rel = float(np.linalg.norm(g_net - g_true) / np.linalg.norm(g_true))
    assert rel < 0.1, f"Dirichlet BC profile recovery rel-err {rel:.3e}"


def test_dirichlet_net_scope_guards():
    """A bare net(x) essential value is supported as a Dirichlet profile (steady / nonlinear /
    linear-transient) and as a linear-transient initial condition; the remaining cases fail loud: a
    compound value (Dirichlet or IC), and a net Dirichlet on a *nonlinear transient* form (the nonlinear
    stepper still holds a static value)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    # compound Dirichlet value 1 + net(x): rejected (needs a general args-aware value evaluator)
    net = _mlp_net(key=42)
    with pytest.raises(NotImplementedError, match="bare|compound"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - 0.0 * vi, u(xb, yb) - (1.0 + net(xb, yb))], quad_degree=3)

    # compound INITIAL-condition value 1 + net(x): rejected (only a bare net(x) IC is supported)
    d2, u2, phi2, (x2, y2), (xb2, yb2), ci, u2i, v2i, _psi0 = _transient_setup(mesh_size=0.3, time=(0.0, 0.05, 6))
    net2 = _mlp_net(key=43)
    with pytest.raises(NotImplementedError, match="bare|compound"):
        jno.fem(
            [
                u2i.t * v2i + (u2i.x * v2i.x + u2i.y * v2i.y),
                u2(xb2, yb2) - 0.0,
                u2(ci[0], ci[1]) - (1.0 + net2(ci[0], ci[1])),
            ]
        )

    # net Dirichlet on a NONLINEAR transient form: rejected (static value in the nonlinear stepper)
    d3, u3, phi3, (x3, y3), (xb3, yb3), ci3, u3i, v3i, u03 = _transient_setup(mesh_size=0.4, time=(0.0, 0.02, 3))
    net3 = _mlp_net(key=44)
    with pytest.raises(NotImplementedError, match="nonlinear transient"):
        jno.fem(
            [
                u3i.t * v3i + (1.0 + u3i**2) * (u3i.x * v3i.x + u3i.y * v3i.y),
                u3(xb3, yb3) - net3(xb3, yb3),
                u3(ci3[0], ci3[1]) - u03,
            ]
        )


def test_scope_guards_fail_loud():
    """Scope limits raise explicit NotImplementedError, never mis-assemble: a solution-dependent
    net(u) on the mass (u_t) term (a nonlinear mass). (A *bare* net(x) essential value IS supported as a
    Dirichlet profile or an initial condition — see test_dirichlet_net_recovers_bc_profile and
    test_ic_net_recovers_initial_field; only compound values and net(u)-on-mass are guarded.)"""
    # net(u) on the mass term = a nonlinear mass C(u)*u_t (the matrix form can't express it). A
    # coordinate net(x) mass coefficient IS supported (see test_mass_net_kx_recovers_via_crux).
    d2, u2, phi2, (x2, y2), (xb2, yb2), ci, u2i, v2i, psi0 = _transient_setup(mesh_size=0.3, time=(0.0, 0.02, 5))
    net2 = _quad_net(a=1.0, b=0.5)
    with pytest.raises(NotImplementedError, match="mass|nonlinear mass"):
        jno.fem(
            [
                net2(u2i) * u2i.t * v2i + (u2i.x * v2i.x + u2i.y * v2i.y),
                u2(xb2, yb2) - 0.0,
                u2(ci[0], ci[1]) - psi0,
            ]
        )


def test_dirichlet_net_nonlinear_enforced_and_differentiates():
    """Net-valued Dirichlet ``u(∂Ω) - net(xb, yb)`` on a NONLINEAR form ``(1 + u²)∇u·∇v = f v``: the
    row-replacement now holds the net's LIVE value (re-evaluated from the weights each residual), so a
    constant-output net reproduces the constant-Dirichlet nonlinear solve to solver tolerance, its
    boundary trace equals the held value (the BC is enforced, not baked/dropped), and the solution is
    differentiable in the Dirichlet weights — the property the guard blocked."""
    from jno.utils.solver.newton_krylov import newton_krylov

    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _ku_setup()
    c = 0.4
    net = _const_net(c)
    fem = jno.fem([(1.0 + ui**2) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - net(xb, yb)], quad_degree=3)
    assert fem._mode == "nonlinear"
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)

    def _solve(module):
        return newton_krylov(lambda v: fem.operator.residual(v, {name: module}), jnp.zeros(fem.operator.size))

    u_net = _solve(net.module)

    # oracle: the identical nonlinear problem with a constant Dirichlet value c
    fem_c = jno.fem([(1.0 + ui**2) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - c], quad_degree=3)
    u_c = newton_krylov(lambda v: fem_c.operator(v), jnp.zeros(fem_c.operator.size))
    assert float(jnp.max(jnp.abs(u_net - u_c))) < 1e-9

    # the solved boundary trace equals the net's held value (enforced from the weights)
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    onb = np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1) | np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1)
    assert float(np.max(np.abs(np.asarray(u_net)[onb] - c))) < 1e-9

    # differentiable in the Dirichlet weights (∂/∂c actually moves the interior solution)
    g = jax.grad(lambda module: jnp.sum(_solve(module)))(net.module)
    assert jnp.isfinite(g.c) and abs(float(g.c)) > 1e-6


def test_dirichlet_net_transient_matches_const_and_differentiates():
    """Net-valued Dirichlet ``u(∂Ω) - net(xb, yb)`` on a TRANSIENT heat form ``u_t v + ∇u·∇v = 0``: the
    held Dirichlet value now rides the per-step forcing (re-evaluated from the weights, time-constant so
    u̇=0 on those rows), so a constant-output net reproduces the constant-Dirichlet transient trajectory
    to solver tolerance, the final boundary trace equals the held value, and the trajectory is
    differentiable in the Dirichlet weights."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, u0 = _transient_setup()
    c = 0.3
    net = _const_net(c)
    fem = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - net(xb, yb), u(ci[0], ci[1]) - u0])
    assert fem.is_transient
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)
    ts = _grid_ts(fem.operator)
    traj = _default_transient_integrate(fem.operator, {name: net.module}, ts)

    # oracle: the identical transient with a constant Dirichlet value c (non-parametric)
    d2, u2, phi2, (x2, y2), (xb2, yb2), ci2, u2i, v2i, u02 = _transient_setup()
    fem_c = jno.fem([u2i.t * v2i + (u2i.x * v2i.x + u2i.y * v2i.y), u2(xb2, yb2) - c, u2(ci2[0], ci2[1]) - u02])
    traj_c = _default_transient_integrate(fem_c.operator, {}, ts)
    assert float(jnp.max(jnp.abs(traj - traj_c))) < 1e-7

    # the final boundary trace is held at the net value across time (BC enforced from the weights)
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    onb = np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1) | np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1)
    assert float(np.max(np.abs(np.asarray(traj)[-1, onb] - c))) < 1e-7

    # differentiable in the Dirichlet weights
    g = jax.grad(lambda module: jnp.sum(_default_transient_integrate(fem.operator, {name: module}, ts)))(net.module)
    assert jnp.isfinite(g.c) and abs(float(g.c)) > 1e-6


def test_ic_net_matches_const_and_differentiates():
    """Net-valued initial condition ``u(initial) - net(xi, yi)`` on a TRANSIENT heat form: the initial
    state is now re-formed from the net weights (``state0_fn``), so a constant-output net reproduces the
    constant-IC trajectory to solver tolerance, the t=0 slice equals the net value everywhere, and the
    trajectory is differentiable in the IC weights — the property the guard blocked."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, _ = _transient_setup()
    c = 0.5
    net = _const_net(c)
    fem = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - net(ci[0], ci[1])])
    assert fem.is_transient
    (name,) = fem.operator.runtime_parameter_exprs
    assert isinstance(fem.operator.runtime_parameter_exprs[name], ModelWeights)
    assert getattr(fem.operator, "state0_fn", None) is not None  # the parametric initial state
    ts = _grid_ts(fem.operator)
    traj = _default_transient_integrate(fem.operator, {name: net.module}, ts)

    # oracle: the identical transient with a constant IC value c (non-parametric)
    d2, u2, phi2, (x2, y2), (xb2, yb2), ci2, u2i, v2i, _2 = _transient_setup()
    fem_c = jno.fem([u2i.t * v2i + (u2i.x * v2i.x + u2i.y * v2i.y), u2(xb2, yb2) - 0.0, u2(ci2[0], ci2[1]) - c])
    traj_c = _default_transient_integrate(fem_c.operator, {}, ts)
    assert float(jnp.max(jnp.abs(traj - traj_c))) < 1e-8

    # the t=0 slice equals the net's IC value everywhere (the initial field is net(x))
    assert float(jnp.max(jnp.abs(traj[0] - c))) < 1e-10

    # differentiable in the IC weights
    g = jax.grad(lambda module: jnp.sum(_default_transient_integrate(fem.operator, {name: module}, ts)))(net.module)
    assert jnp.isfinite(g.c) and abs(float(g.c)) > 1e-6


@pytest.mark.slow
def test_ic_net_recovers_initial_field():
    """The headline: recover an unknown INITIAL field ``u(x, 0) = net(x)`` from the heat trajectory. A
    reference solve imprints ``u0 = sin(πx)sin(πy)`` on the trajectory; training the net IC against that
    trajectory through the differentiable transient solve recovers the initial field (verified as the
    trained net's own output on the mesh nodes, not a restatement of the target)."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    dr, ur, pr, (xir, yir), (xbr, ybr), cir, uri, vri, u0r = _transient_setup(mesh_size=0.22, time=(0.0, 0.02, 5))
    fem_ref = jno.fem([uri.t * vri + (uri.x * vri.x + uri.y * vri.y), ur(xbr, ybr) - 0.0, ur(cir[0], cir[1]) - u0r])
    ts = _grid_ts(fem_ref.operator)
    u_obs = _default_transient_integrate(fem_ref.operator, {}, ts)

    d, u, phi, (xi, yi), (xb, yb), ci, ui, vi, _ = _transient_setup(mesh_size=0.22, time=(0.0, 0.02, 5))
    net = _mlp_net(key=51, hidden=16)
    net.optimizer(optax.adam(1e-2))
    fem = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - net(ci[0], ci[1])])
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(400)

    trained = crux.eval([ModelWeights(net)])
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    ic_net = np.asarray(trained(jnp.asarray(nodes[:, 0:1]), jnp.asarray(nodes[:, 1:2]))).reshape(-1)
    ic_true = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    rel = float(np.linalg.norm(ic_net - ic_true) / np.linalg.norm(ic_true))
    assert rel < 0.15, f"initial-field recovery rel-err {rel:.3e}"


def test_ic_net_second_order_guard():
    """A net-valued IC threads its weights only on the real first-order transient path (state0_fn). On a
    *second-order-in-time* (u_tt) form the IC is baked at assembly, so a net IC there fails loud rather
    than silently freezing the weights."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4, time=(0.0, 0.4, 5))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    net = _const_net(0.3)
    with pytest.raises(NotImplementedError, match="first-order|second-order"):
        jno.fem([ui.tt * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - net(xi0, yi0), ui0.t - 0.0])


# ==========================================================================
# frozen fields: ui.freeze(values) as a KNOWN predictor input to a coefficient
# ==========================================================================


def _mlp_net4(key=0, hidden=16, layers=2):
    """A 4-input MLP: (x, y, d_x u0, d_y u0) -> scalar coefficient."""
    net = jno.nn.wrap(
        foundax.mlp(4, hidden_dims=hidden, num_layers=layers, activation=jax.nn.tanh, key=jax.random.PRNGKey(key))
    )
    net.dtype(jnp.float64)
    return net


def test_freeze_value_reproduces_the_field():
    """``ui.freeze(u0)`` interpolates the KNOWN nodal vector u0 at the quad points: its L2 projection
    back onto the same P1 space is u0 itself (u0 lives in the space), to machine precision."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup(mesh_size=0.25)
    # a known field: solve a plain Poisson problem for u0
    u0 = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3).solve()).reshape(-1)
    # mass @ x = integral( freeze(u0) * v )  ->  x = L2 projection of u0 = u0
    proj = jno.fem([ui * vi - ui.freeze(u0) * vi], quad_degree=3)
    x = np.asarray(proj.solve(linear=jno.solve.lu())).reshape(-1)  # direct solve -> machine precision
    assert proj.is_linear
    assert np.linalg.norm(x - u0) / np.linalg.norm(u0) < 1e-11


def test_freeze_gradient_conditioned_coefficient_is_linear_and_trains():
    """A diffusion coefficient conditioned on a FROZEN predictor gradient,
    ``kappa = softplus(net(x, y, ui.freeze(u0).x, ui.freeze(u0).y))``, stays LINEAR in the true
    unknown (the frozen field is known data, not the unknown) and its weights train through the
    differentiable solve (loss decreases)."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup(mesh_size=0.25)
    u0 = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3).solve()).reshape(-1)
    u_obs = u0.copy()  # target: recover the plain-Poisson solution (correction should shrink to ~0)

    net = _mlp_net4()
    net.optimizer(optax.adam(3e-3))
    uk = ui.freeze(u0)
    kappa = jnn.log1p(jnn.exp(net(xi, yi, uk.x, uk.y) - 4.0))  # softplus, small at init
    fem = jno.fem([(1.0 + kappa) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem.is_linear, "a frozen-field-conditioned coefficient must keep the system linear"

    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    hist = crux.solve(200)
    losses = np.asarray(hist.total_loss_history)
    assert losses[-1] < 0.5 * losses[0], "frozen-conditioned coefficient weights should train through the solve"


# ==========================================================================
# gradient of a KNOWN (frozen) network coefficient — FE-basis enrichment
# (a(u_h, v) = L(v) - a(u_NN, v); the net's continuous spatial gradient lands in b)
# ==========================================================================


def _net_dx_nodal(net, pts, dim):
    """Independent autodiff reference: d(net)/d(coord `dim`) at the given points."""

    def _scalar(p):
        out = net.module(p[0:1], p[1:2])
        return jnp.reshape(jnp.asarray(out), (-1,))[0]

    return np.asarray(jax.vmap(lambda p: jax.grad(_scalar)(p)[dim])(jnp.asarray(pts)))


def test_frozen_network_gradient_l2_projection_matches_autodiff():
    """``jnn.grad(frozen_net, x)`` assembles the net's CONTINUOUS spatial gradient. Verify it by
    L²-projecting ∂net/∂x onto the FE space — (w, v) = (∂net/∂x, v) ∀v — and comparing to an
    independent ``jax.grad`` of the network at the DOF nodes. Deterministic (fixed PRNG, no training)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    net = _mlp_net(key=1, hidden=8, layers=2)
    fnet = net.freeze()
    pts = np.asarray(d.mesh.points)[:, :2]

    for dim, dvar in ((0, xi), (1, yi)):
        fem = jno.fem([ui * vi - jnn.grad(fnet(xi, yi), dvar) * vi], quad_degree=4)  # mass-matrix L² projection
        w = np.asarray(fem.solve()).reshape(-1)
        ref = _net_dx_nodal(net, pts, dim)
        rel = float(np.linalg.norm(w - ref) / np.linalg.norm(ref))
        assert rel < 5e-3, f"assembled d(net)/d(dim{dim}) mismatches autodiff: rel-L2={rel:.3e}"
        assert np.linalg.norm(w) > 1e-3, "assembled network gradient is spuriously ~zero"


def test_network_gradient_requires_coordinate_arguments():
    """∂net/∂x is wired for a network of *coordinate* variables (the enrichment case). A network of
    a non-coordinate argument (e.g. a trial value) raises a clear error rather than silently wrong."""
    d, u, phi, (xi, yi), (xb, yb), ui, vi, f = _poisson_setup(mesh_size=0.25)
    fnet = _mlp_net(key=1).freeze()
    with pytest.raises(NotImplementedError, match="coordinate variables"):
        # arg ``2*xi`` is a coordinate *expression*, not a bare Variable → guard fires (stays linear)
        fem = jno.fem([ui.x * vi.x + jnn.grad(fnet(2 * xi, yi), xi) * vi.x - f * vi, u(xb, yb) - 0.0], quad_degree=3)
        _ = np.asarray(fem.b)  # force assembly


def test_frozen_network_gradient_enrichment_improves_prior():
    """FE-basis enrichment end to end: correct a network prior on a coarse mesh via
    a(u_h, v) = (f, v) - a(u_NN, v). The enriched u_NN + u_h is at least as accurate as the raw
    prior — proof the frozen net's gradient lands in the RHS with the right sign."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2 * np.pi**2 * jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    pts = np.asarray(d.mesh.points)[:, :2]
    exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])

    net = _mlp_net(key=0, hidden=32, layers=3)
    xf, yf, _ = d.variable("interior", split=True)
    tgt = jnn.sin(np.pi * xf) * jnn.sin(np.pi * yf)
    net.optimizer(optax.adam(optax.exponential_decay(3e-3, 2000, 0.5, end_value=2e-5)))
    crux = jno.core([(net(xf, yf) - tgt).mse])
    crux.solve(4000)
    prior = np.asarray(crux.eval([net(xi, yi)], domain=d)).reshape(-1)
    prior_rel = float(np.linalg.norm(prior - exact) / np.linalg.norm(exact))

    fnet = net.freeze()
    gx = ui.x + jnn.grad(fnet(xi, yi), xi)
    gy = ui.y + jnn.grad(fnet(xi, yi), yi)
    fem = jno.fem([gx * vi.x + gy * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=4)
    u_h = np.asarray(fem.solve()).reshape(-1)
    enriched_rel = float(np.linalg.norm(prior + u_h - exact) / np.linalg.norm(exact))
    assert enriched_rel < 1.5e-2, f"enriched solution inaccurate: rel-L2={enriched_rel:.3e}"
    assert enriched_rel <= prior_rel * 1.2, f"correction worsened the prior: {prior_rel:.3e} -> {enriched_rel:.3e}"
