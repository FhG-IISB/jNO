"""Neural-network + FEM hybrids that had no test: each one trained or evaluated end to end, with an oracle.

1. **Solver-in-the-loop / learned correction** (Um, Brand, Fei, Holl & Thuerey, NeurIPS 2020): a coarse
   FEM model missing a physical term, corrected by a network trained THROUGH the coarse ``fem.solve()``.
2. **Operator learning on FEM data** (DeepONet: Lu, Jin, Pang, Zhang & Karniadakis, Nat. Mach. Intell.
   3 (2021) 218): a parametric family solved with ``jno.fem``, a DeepONet trained on it, scored on
   held-out parameters.
3. **Physics-informed operator with a FEM residual loss** (Gao, Zahr & Wang, CMAME 390 (2022) 114502;
   Wang, Wang & Perdikaris, Sci. Adv. 7 (2021) eabi8605): the same map learned from the assembled FE
   residual alone, no solution data.
4. **Network trial + inverse parameter** (hp-VPINN, Kharazmi, Zhang & Karniadakis, CMAME 374 (2021)
   113547): a VPINN field and an unknown coefficient recovered together.
5. **Deep Ritz** (E & Yu, Commun. Math. Stat. 6 (2018) 1): the energy functional of an exact network.
6. **Network warm start for Newton**: a trained surrogate as the ``x0=`` of a Newton solve.

Three of the tests below expose jNO defects found while writing the others and are left failing (they
assert the correct behaviour): a linear parametric ``fem.solve(k=v)`` that ignores ``v``, a per-node
data tensor that is silently cut to its first node, and a network trial that cannot be evaluated on a
batched (``N * domain``) operator-learning domain.

Run with x64 (assembly runs in float64).
"""

import functools

import numpy as np
import pytest

pytest.importorskip("foundax", reason="foundax required for the MLP / DeepONet networks")

import equinox as eqx  # noqa: E402
import foundax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402
from jno.trace import ModelWeights  # noqa: E402

grad, inner = jno.np.grad, jno.np.inner
sin = jno.np.sin
PI = np.pi

# A loss built from a global FEM solve has no spatial Variable, so crux needs an explicit domain.
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64; set x64 per-test with save/restore."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _unit_square(h):
    d = jno.shape.rect(0, 0, 1, 1, size=h).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, u, phi, (xi, yi), (xb, yb)


def _rel(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _scalar_param(name, value=0.0):
    p = jno.np.parameter((1,), name=name, key=jax.random.PRNGKey(0))
    p.dtype(jnp.float64)
    p.initialize(jax.nn.initializers.constant(value))
    return p


def _mlp(n_in, key, hidden=16, layers=2):
    net = jno.nn.wrap(
        foundax.mlp(n_in, hidden_dims=hidden, num_layers=layers, activation=jax.nn.tanh, key=jax.random.PRNGKey(key))
    )
    net.dtype(jnp.float64)
    return net


def _deeponet(n_params, key=0, basis=32, hidden=64, layers=3):
    net = jno.nn.wrap(
        foundax.deeponet(
            n_sensors=n_params,  # branch input: the parameter vector
            coord_dim=2,  # trunk input: a node coordinate
            basis_functions=basis,
            hidden_dim=hidden,
            n_layers=layers,
            activation=jnp.tanh,
            key=jax.random.PRNGKey(key),
        )
    )
    net.dtype(jnp.float64)
    return net


def _node_batch(nodes, params):
    """``B`` copies of the FE node cloud (the operator-learning layout), one parameter row per copy.

    Returns the domain, the per-sample parameter tag and the trunk input ``concat([x, y])``."""
    dom = len(params) * jno.domain.from_array({"nodes": np.asarray(nodes)})
    x, y, _ = dom.variable("nodes")
    p = dom.variable("p", np.asarray(params)[:, None, :])  # (B, 1, n_params): one row per sample
    return dom, p, jno.np.concat([x, y], axis=-1)


def _predict(crux, net, nodes, params):
    """The trained operator's own output at ``params`` (any rows, seen or not), shape ``(B, n_nodes)``."""
    dom, p, xy = _node_batch(nodes, params)
    return np.asarray(crux.eval([net(p, xy)], domain=dom)).reshape(len(params), -1)


# ==========================================================================
# 1. solver-in-the-loop: a learned closure trained through the coarse solve
# ==========================================================================


def test_solver_in_the_loop_learned_closure_corrects_coarse_model():
    """Solver-in-the-loop (Um, Brand, Fei, Holl & Thuerey, NeurIPS 2020) on a steady problem.

    Truth: ``-Δu + u³ = f`` with ``u* = 2 sin(πx) sin(πy)``. The coarse P1 model omits the cubic term;
    the correction is a state-dependent closure ``c(u) = net(u)`` placed in the coarse weak form, and the
    net is trained THROUGH the coarse nonlinear ``fem.solve()`` so the coarse solution matches ``u*`` at
    its nodes. Checked:

    * on the training forcing the error to ``u*`` drops by >= 20x against the uncorrected coarse model
      (measured 0.114 -> 0.00136, 84x);
    * on a HELD-OUT forcing of a different shape (``u* = 1.5 * 16 x(1-x) y(1-y)``, same value range) the
      frozen closure still cuts the error >= 8x (measured 0.068 -> 0.0031, 22x) and lands within 2x of a
      coarse model that has the exact physics (measured 1.14x) -- the correction learned a function of
      the state, not of the position;
    * the learned term approximates the missing one where it matters: ``net(u*) ≈ u*³`` at the training
      field's nodes to < 25 % relative L2 (measured 12 %). Near ``u = 0`` (the Dirichlet edge) the closure
      is barely probed and is not checked pointwise.
    """
    d, u, phi, (xi, yi), (xb, yb) = _unit_square(0.1)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    X = [xi, yi]
    stiffness = inner(grad(u, X), grad(phi, X), 1)
    bc = u(xb, yb) - 0.0

    s = sin(PI * xi) * sin(PI * yi)
    bubble = 16.0 * xi * (1 - xi) * yi * (1 - yi)
    f_train = 2.0 * 2 * PI**2 * s + (2.0 * s) ** 3  # -Δu* + u*³ for u* = 2 s
    f_heldout = 1.5 * 32.0 * (xi * (1 - xi) + yi * (1 - yi)) + (1.5 * bubble) ** 3  # u* = 1.5 bubble

    fem_uncorrected = jno.fem([stiffness - f_train * vi, bc])
    nodes = np.asarray(fem_uncorrected.points)
    s_n = np.sin(PI * nodes[:, 0]) * np.sin(PI * nodes[:, 1])
    b_n = 16.0 * nodes[:, 0] * (1 - nodes[:, 0]) * nodes[:, 1] * (1 - nodes[:, 1])
    u_train, u_heldout = 2.0 * s_n, 1.5 * b_n

    net = _mlp(1, key=0)
    net.optimizer(optax.adam(1e-2))
    fem_coarse = jno.fem([stiffness + net(ui) * vi - f_train * vi, bc])
    assert fem_coarse.mode == "nonlinear"
    u_node = fem_coarse.solve()
    crux = jno.core([(u_node - jnp.asarray(u_train)).mse], domain=_DUMMY)
    crux.solve(2000)

    err_before = _rel(fem_uncorrected.solve(), u_train)
    err_after = _rel(crux.eval([u_node]), u_train)
    assert err_after < err_before / 20.0, f"training forcing: {err_before:.3e} -> {err_after:.3e}"

    # the frozen closure on a forcing it never saw
    from jno.utils.solver.newton_krylov import newton_krylov

    closure = crux.eval([ModelWeights(net)])
    fem_h = jno.fem([stiffness + net(ui) * vi - f_heldout * vi, bc])
    (name,) = fem_h.operator.runtime_parameter_exprs
    u_h = newton_krylov(lambda v: fem_h.operator.residual(v, {name: closure}), jnp.zeros(fem_h.operator.size))
    err_h_before = _rel(jno.fem([stiffness - f_heldout * vi, bc]).solve(), u_heldout)
    err_h_exact_physics = _rel(jno.fem([stiffness + ui**3 * vi - f_heldout * vi, bc]).solve(), u_heldout)
    err_h_after = _rel(u_h, u_heldout)
    assert err_h_after < err_h_before / 8.0, f"held-out forcing: {err_h_before:.3e} -> {err_h_after:.3e}"
    assert err_h_after < 2.0 * err_h_exact_physics, f"{err_h_after:.3e} vs exact-physics coarse {err_h_exact_physics:.3e}"

    learned = np.asarray(closure(jnp.asarray(u_train).reshape(-1, 1))).reshape(-1)
    assert _rel(learned, u_train**3) < 0.25, f"closure vs u³ at the nodes: {_rel(learned, u_train**3):.3e}"


# ==========================================================================
# 2. operator learning on a FEM-generated dataset
# ==========================================================================

_P_LO, _P_HI = np.array([0.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0])


def _family_form(d, u, phi, xi, yi, xb, yb, p1, p2, p3):
    """``-∇·((1 + p1 x) ∇u) = 10 (1 + p2 sin(πy) + p3 x)``, ``u = 0`` on the boundary."""
    X = [xi, yi]
    vi = phi.bind(x=xi, y=yi)
    k = 1.0 + p1 * xi
    f = 10.0 * (1.0 + p2 * sin(PI * yi) + p3 * xi)
    return [k * inner(grad(u, X), grad(phi, X), 1) - f * vi, u(xb, yb) - 0.0]


@functools.lru_cache(maxsize=1)
def _poisson_family():
    """ONE parametric form, solved over 32 training + 8 held-out parameter draws with the public sweep
    driver ``fem.solve(continuation=...)``. Shared by the data-driven and the residual-driven operator tests."""
    d, u, phi, (xi, yi), (xb, yb) = _unit_square(0.1)
    p1, p2, p3 = (_scalar_param(n) for n in ("p1", "p2", "p3"))
    fem = jno.fem(_family_form(d, u, phi, xi, yi, xb, yb, p1, p2, p3))
    rng = np.random.default_rng(0)
    P_train = _P_LO + (_P_HI - _P_LO) * rng.random((32, 3))
    P_test = _P_LO + (_P_HI - _P_LO) * rng.random((8, 3))

    def sweep(P):
        cont = jno.solve.continuation("all", p1=P[:, 0], p2=P[:, 1], p3=P[:, 2])
        return np.asarray(fem.solve(continuation=cont)).reshape(len(P), -1)

    return fem, np.asarray(fem.points), P_train, sweep(P_train), P_test, sweep(P_test)


def test_deeponet_on_fem_data_generalises_to_heldout_parameters():
    """Operator learning on FEM data (DeepONet, Lu et al., Nat. Mach. Intell. 3 (2021) 218).

    A 3-parameter family ``-∇·((1 + p1 x)∇u) = 10(1 + p2 sin(πy) + p3 x)`` is solved with ``jno.fem`` at 32
    random parameter draws; a DeepONet (branch = the parameters, trunk = the node coordinate) is trained
    through ``jno.core`` on a plain data loss. Oracle, on 8 HELD-OUT draws: mean relative L2 against the
    FEM solutions < 0.12 (measured 0.068) and < 0.35x the error of predicting the training mean
    (measured 0.34). The dataset itself is checked first: one swept sample equals a form rebuilt with
    those numbers baked in.
    """
    fem, nodes, P_train, U_train, P_test, U_test = _poisson_family()

    d, u, phi, (xi, yi), (xb, yb) = _unit_square(0.1)
    p = P_test[3]
    rebuilt = np.asarray(jno.fem(_family_form(d, u, phi, xi, yi, xb, yb, *map(float, p))).solve()).reshape(-1)
    assert np.abs(rebuilt - U_test[3]).max() < 1e-7 * np.abs(rebuilt).max()

    steps = 2000
    net = _deeponet(3)
    net.optimizer(optax.adam(optax.cosine_decay_schedule(3e-3, steps, alpha=1e-2)))
    dom, p_tag, xy = _node_batch(nodes, P_train)
    u_data = dom.variable("u_data", U_train[:, None, :, None])  # (B, T=1, n_nodes, 1)
    crux = jno.core([(net(p_tag, xy) - u_data).mse])
    crux.solve(steps)

    rel = lambda A, B: np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)  # noqa: E731
    err = rel(_predict(crux, net, nodes, P_test), U_test).mean()
    err_mean = rel(np.broadcast_to(U_train.mean(axis=0), U_test.shape), U_test).mean()
    assert err < 0.12, f"held-out DeepONet rel-L2 {err:.3e}"
    assert err < 0.35 * err_mean, f"DeepONet {err:.3e} vs training-mean predictor {err_mean:.3e}"


def test_per_node_training_tensor_reaches_the_loss_whole():
    """Operator-learning targets are one value per node per sample, ``(B, n_nodes, 1)``. Attached to a
    steady ``B * domain`` it must reach the loss whole, or be refused -- never cut down silently.

    FAILS today: it arrives as ``(B, 1)``, the FIRST node of each sample, with no error, so a data loss
    on it trains against one number per sample (measured: a DeepONet then scores 1.00 rel-L2 on held-out
    data). ``_normalize_tensor_time_axis`` (jno/domain/domain_class.py:1027) leaves rank < 4 alone on
    the reasoning that the time inference will not touch it -- evidently it still is. ``(B, 1, n, 1)``
    (explicit time axis) arrives intact, and is what the tests above use.
    """
    B, n = 4, 6
    nodes = np.random.default_rng(0).random((n, 2))
    data = np.arange(B * n, dtype=float).reshape(B, n, 1)
    dom = B * jno.domain.from_array({"nodes": nodes})
    dom.variable("nodes")
    try:
        target = dom.variable("u_data", data)
    except (ValueError, NotImplementedError):
        return  # refusing the ambiguous layout is also correct
    got = np.asarray(jno.core([(target * 1.0).mse]).eval([target]))
    assert got.size == data.size, f"(B, n, 1) data arrived as {got.shape}"
    assert np.allclose(got.reshape(B, n), data.reshape(B, n))


# ==========================================================================
# 3. physics-informed operator: the FEM residual as the only loss
# ==========================================================================


def test_physics_informed_operator_from_fem_residual_alone():
    """Physics-informed operator learning with a FEM residual loss (Gao, Zahr & Wang, CMAME 390 (2022)
    114502; Wang, Wang & Perdikaris, Sci. Adv. 7 (2021) eabi8605). NO solution data.

    The DeepONet of the data test maps (p1, p2, p3) to nodal values ``U``; the loss is the assembled FE
    residual ``r = A(p) U - b(p)`` of that output (``fem.residual`` with the parameters as runtime
    ``args``, wrapped with ``jno.fn`` -- jNO vmaps it per sample). The residual is left-preconditioned by
    the fixed operator at the centre of the parameter box, ``A0^-1 r``: its zero set is unchanged, but the
    plain ``|r|²`` squares ``cond(A)`` and reaches only 0.32 held-out rel-L2 in the same 2000 steps.
    Oracle, on the 8 held-out draws: mean relative L2 against the FEM solutions < 0.12 (measured 0.056)
    and < 0.35x the training-mean predictor.
    """
    fem, nodes, P_train, U_train, P_test, U_test = _poisson_family()
    A0, _ = fem.operator.evaluate({"p1": jnp.asarray(1.5), "p2": jnp.asarray(0.0), "p3": jnp.asarray(0.0)})
    A0_inv = jnp.linalg.inv(jnp.asarray(A0.todense()))
    residual = fem.residual

    def preconditioned_residual(U, p):  # per sample: U (n_nodes, 1), p (3,)
        p = jnp.reshape(p, (-1,))
        return A0_inv @ residual(jnp.reshape(U, (-1,)), {"p1": p[0], "p2": p[1], "p3": p[2]})

    steps = 2000
    net = _deeponet(3)
    net.optimizer(optax.adam(optax.cosine_decay_schedule(3e-3, steps, alpha=1e-2)))
    dom, p_tag, xy = _node_batch(nodes, P_train)  # the training PARAMETERS only
    r = jno.fn(preconditioned_residual, [net(p_tag, xy), p_tag], name="fem_residual")
    crux = jno.core([r.mse])
    crux.solve(steps)

    rel = lambda A, B: np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)  # noqa: E731
    err = rel(_predict(crux, net, nodes, P_test), U_test).mean()
    err_mean = rel(np.broadcast_to(U_train.mean(axis=0), U_test.shape), U_test).mean()
    assert err < 0.12, f"held-out residual-trained operator rel-L2 {err:.3e}"
    assert err < 0.35 * err_mean, f"{err:.3e} vs training-mean predictor {err_mean:.3e}"


def test_network_trial_on_batched_domain_gives_per_sample_residual():
    """The OTHER route to a physics-informed operator: a VPINN network trial conditioned on a per-sample
    parameter (docs/fem/formulations.md, "The trial may be a network") needs the weak form evaluated on a
    batched ``N * domain`` -- the operator-learning layout.

    Oracle: with a network that ignores the sample, every sample's residual loss equals the single-domain
    one. FAILS today: ``jno.fem`` builds the GroupedAssembly without complaint, but evaluating it raises
    ``ValueError: Expected shape_vals_flat.ndim == 2, got shape (3,)`` (jno/trace_evaluator.py:2799) --
    the FE shape table ``N_flat`` ``(n_q, 3)`` reaches the per-sample evaluation as one row. Likely
    cause: ``_effective_batch_count`` (jno/domain/domain_class.py:1054) takes the max leading dimension
    over EVERY ``domain.context`` entry, so the lowering's FE tables (``n_q = 198`` rows) read as a
    batch of 198 and are vmapped row by row once the domain is batched.
    """

    def residual_loss(batch):
        base = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
        d = base if batch == 1 else batch * base
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = _mlp(2, key=0)(xi, yi) * xi * (1 - xi) * yi * (1 - yi)
        pde = jno.fem([grad(u_net, xi) * grad(vi, xi) + grad(u_net, yi) * grad(vi, yi) - 10.0 * xi * vi, u(xb, yb) - 0.0])
        return np.asarray(jno.core([pde.mse], domain=d).eval([pde.mse])).reshape(-1)

    single = residual_loss(1)
    batched = residual_loss(4)
    assert np.allclose(batched, single[0], rtol=1e-10), f"batched {batched} vs single {single}"


# ==========================================================================
# 4. network trial + inverse parameter
# ==========================================================================


def test_vpinn_network_trial_recovers_coefficient_with_data():
    """Network trial (hp-VPINN, Kharazmi, Zhang & Karniadakis, CMAME 374 (2021) 113547) and a
    ``jno.np.parameter`` trained in ONE loss: ``-k Δu = f`` with truth ``k = 3``,
    ``u* = sin(πx) sin(πy)``, ``k`` starting at 1. The residual alone is degenerate (any ``k`` with ``u``
    rescaled), so a data term on ``u*`` pins the field (docs/fem/formulations.md, the measured row
    "k: 1.00 -> 2.901"). The data term is weighted 100x: the residual carries ``2π²k ≈ 59`` and, at equal
    weights, the same budget stops at k = 2.69 (4000 steps) / 2.97 (10000). Order-1 test functions, as a
    network trial needs.

    Oracle: recovered ``k`` within 1 % of 3 (measured 3.0004) and the field within 1e-2 relative L2 of
    ``u*`` (measured 4.5e-4).
    """
    d, u, phi, (xi, yi), (xb, yb) = _unit_square(0.2)
    vi = phi.bind(x=xi, y=yi)
    u_true = sin(PI * xi) * sin(PI * yi)
    f = 3.0 * 2 * PI**2 * u_true

    net = _mlp(2, key=0)
    net.optimizer(optax.adam(1e-2))
    k = _scalar_param("k", 1.0)
    k.optimizer(optax.adam(5e-2))
    u_net = 16.0 * net(xi, yi) * xi * (1 - xi) * yi * (1 - yi)  # hard-BC ansatz
    pde = jno.fem([k * (grad(u_net, xi) * grad(vi, xi) + grad(u_net, yi) * grad(vi, yi)) - f * vi, u(xb, yb) - 0.0])

    crux = jno.core([pde.mse, 100.0 * (u_net - u_true).mse], domain=d)
    crux.solve(4000)

    k_hat = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    field_err = _rel(crux.eval([u_net]), crux.eval([u_true]))
    assert abs(k_hat - 3.0) < 0.03, f"recovered k = {k_hat:.4f}, truth 3"
    assert field_err < 1e-2, f"field rel-L2 {field_err:.3e}"


# ==========================================================================
# 5. Deep Ritz: the energy of an exact network, and the Ritz minimiser
# ==========================================================================


class _ScaledBubble(eqx.Module):
    """``c · x(1-x) y(1-y)``, elementwise (jNO calls it on (N, 1) arrays and per point for derivatives).
    At ``c = 1`` it is the exact solution of ``-Δu = 2[x(1-x) + y(1-y)]`` with ``u = 0`` on the square."""

    c: jnp.ndarray

    def __call__(self, x, y):
        return self.c * x * (1 - x) * y * (1 - y)


def _ritz_energy(c, quadrature):
    d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
    xi, yi, _ = d.variable("interior", split=True)
    net = jno.nn.wrap(_ScaledBubble(c=jnp.asarray(float(c))))
    net.dtype(jnp.float64)
    uu = net(xi, yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    energy = (0.5 * (grad(uu, xi) ** 2 + grad(uu, yi) ** 2) - f * uu).integrate(quadrature=quadrature)
    return d, net, energy


def test_deep_ritz_energy_of_exact_network_matches_analytic():
    """Deep Ritz (E & Yu, Commun. Math. Stat. 6 (2018) 1), training-free: the energy
    ``J[u] = ∫ ½|∇u|² - f u`` of the network ``c · x(1-x)y(1-y)`` is ``c²/90 - c/45`` analytically, a
    parabola whose minimum ``-1/90`` sits at the exact solution ``c = 1``. The integrand is a degree-6
    polynomial, so ``.integrate(quadrature=6)`` must reproduce it to round-off at every ``c`` -- which
    also pins the linear term (the Ritz stationarity ``a(u*, v) = (f, v)``) and the network-gradient path
    (measured |ΔJ| <= 2e-17)."""
    for c in (0.5, 1.0, 2.0):
        d, _, energy = _ritz_energy(c, quadrature=6)
        J = float(np.asarray(jno.core([energy], domain=d).eval([energy])).reshape(-1)[0])
        assert abs(J - (c**2 / 90.0 - c / 45.0)) < 1e-14, f"J(c={c}) = {J!r}"


def test_deep_ritz_training_reaches_the_ritz_minimiser():
    """Deep Ritz trained through ``jno.core`` on a one-parameter trial ``c · x(1-x)y(1-y)``: minimising
    the (signed) energy must drive ``c`` to the Ritz-Galerkin value ``(f, φ)/a(φ, φ) = 1`` -- the exact
    solution, since φ spans it. Oracle: ``|c - 1| < 1e-3`` from ``c = 0.3``."""
    d, net, energy = _ritz_energy(0.3, quadrature=6)
    net.optimizer(optax.adam(5e-2))
    crux = jno.core([energy], domain=d)
    crux.solve(400)
    c = float(np.asarray(crux.eval([ModelWeights(net)]).c).reshape(-1)[0])
    assert abs(c - 1.0) < 1e-3, f"Ritz coefficient {c:.6f}, expected 1"


# ==========================================================================
# 6. a trained surrogate as the Newton warm start
# ==========================================================================


def test_network_warm_start_cuts_newton_steps_to_the_same_solution():
    """Network warm start for Newton (docs/solvers.md, "A neural surrogate is a poor warm start" -- that
    measurement is for Krylov; this is the Newton case). ``-Δu + u³ = 20 a sin(πx) sin(πy)``: a DeepONet
    is trained on 7 FEM solves at ``a ∈ [1, 4]`` (generated by ``fem.solve(a=v)``, which honours the value
    on this NONLINEAR form), then its prediction at the held-out ``a = 2.3`` is the ``x0=`` of a
    sparse-direct Newton solve.

    Oracle: the warm and cold solves converge to the same field (max difference < 1e-7, measured 1.3e-9,
    both inside the driver's own bound), and the warm start takes STRICTLY fewer Newton steps (measured
    2 against 4; the nearest stored snapshot takes 3) -- strict, so an ignored ``x0`` cannot pass.
    """
    d, u, phi, (xi, yi), (xb, yb) = _unit_square(0.1)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    X = [xi, yi]
    stiffness = inner(grad(u, X), grad(phi, X), 1)
    bc = u(xb, yb) - 0.0
    shape = 20.0 * sin(PI * xi) * sin(PI * yi)
    newton = jno.solve.newton(direct=True)

    a = _scalar_param("a", 1.0)
    fem = jno.fem([stiffness + ui**3 * vi - a * shape * vi, bc])
    a_train = np.linspace(1.0, 4.0, 7)
    U_train = np.stack([np.asarray(fem.solve(a=v, nonlinear=newton)).reshape(-1) for v in a_train])
    nodes = np.asarray(fem.points)

    steps = 1000
    net = _deeponet(1, basis=16, hidden=32, layers=2)
    net.optimizer(optax.adam(optax.cosine_decay_schedule(1e-2, steps, alpha=1e-2)))
    dom, a_tag, xy = _node_batch(nodes, a_train[:, None])
    u_data = dom.variable("u_data", U_train[:, None, :, None])
    crux = jno.core([(net(a_tag, xy) - u_data).mse])
    crux.solve(steps)
    guess = _predict(crux, net, nodes, np.array([[2.3]])).reshape(-1)

    fem_h = jno.fem([stiffness + ui**3 * vi - 2.3 * shape * vi, bc])
    cold = np.asarray(fem_h.solve(nonlinear=newton)).reshape(-1)
    st_cold = fem_h.stats["nonlinear"]
    warm = np.asarray(fem_h.solve(x0=jnp.asarray(guess), nonlinear=newton)).reshape(-1)
    st_warm = fem_h.stats["nonlinear"]

    assert st_cold["converged"] and st_warm["converged"]
    assert np.abs(warm - cold).max() < 1e-7, f"warm/cold differ by {np.abs(warm - cold).max():.3e}"
    assert st_warm["steps"] < st_cold["steps"], f"Newton steps: warm {st_warm['steps']} vs cold {st_cold['steps']}"
