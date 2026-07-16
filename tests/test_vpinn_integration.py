"""Lightweight tests for VPINN / weak-form integration."""

import pytest

pytest.importorskip("foundax", reason="foundax required for neural VPINN tests")

import foundax
import jax

import jno
import jno.jnp_ops as jnn
from jno.trace import dump_tree

# ============================================================
# Helpers
# ============================================================


def make_domain(mesh_size=0.35):
    """Create a small rectangular domain for fast VPINN tests."""
    return jno.Shape.rect(0, 0, 1, 1, size=mesh_size).domain()


def init_vpinn_fem(dom, with_neumann_tags=True):
    """
    Initialize FEM quadrature tags used by the VPINN route.

    VPINN uses the same sampled tags as the FEM route:
    - fem_gauss
    - gauss_<boundary_tag>
    """
    bcs = [dom.dirichlet("left")]
    if with_neumann_tags:
        bcs.append(dom.neumann(["right", "top"]))

    # Native FEM context: the same quadrature / shape-function / boundary tensors the
    # grouped-weak-form evaluator reads, built from the native Lagrange + facet machinery.
    dom.init_fem_native(
        element_type="TRI3",
        quad_degree=2,
        bcs=bcs,
    )
    return dom


def make_scalar_net():
    key = jax.random.PRNGKey(0)
    return jnn.nn.wrap(
        foundax.mlp(
            2,
            hidden_dims=16,
            num_layers=2,
            activation=jax.nn.tanh,
            key=key,
        )
    )


def make_vector_net():
    key = jax.random.PRNGKey(0)
    return jnn.nn.wrap(
        foundax.mlp(
            2,
            hidden_dims=16,
            num_layers=2,
            activation=jax.nn.tanh,
            key=key,
            output_dim=2,
        )
    )


# ============================================================
# VPINN variable / tag access
# ============================================================


class TestVpinnVariables:
    def test_fem_gauss_and_boundary_quadrature_variables_exist(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=True)

        xg, yg, _ = dom.variable("fem_gauss", split=True)
        xr, yr, _ = dom.variable("gauss_right", split=True)
        xt, yt, _ = dom.variable("gauss_top", split=True)

        assert xg is not None
        assert yg is not None
        assert xr is not None
        assert yr is not None
        assert xt is not None
        assert yt is not None

    def test_boundary_quadrature_tags_are_created(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=True)

        assert "fem_gauss" in dom._mesh_pool
        assert "gauss_right" in dom._mesh_pool
        assert "gauss_top" in dom._mesh_pool

        assert dom._mesh_pool["fem_gauss"].shape[0] > 0
        assert dom._mesh_pool["gauss_right"].shape[0] > 0
        assert dom._mesh_pool["gauss_top"].shape[0] > 0


# ============================================================
# Scalar VPINN assembly
# ============================================================


class TestVpinnScalarAssembly:
    def test_scalar_volume_weak_form_assembles(self):
        # Authored through jno.fem (the sole entry): the network trial u=net(x,y) sits inside the
        # weak form (detected as a ModelCall) and the Dirichlet condition masks the boundary test
        # functions. jno.fem builds the native fem_context internally (no init_fem / weak.assemble).
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = make_scalar_net()(xi, yi)
        weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - (1.0 + 0.0 * xi) * vi

        pde = jno.fem([weak, u(xb, yb) - 0.0])

        assert pde is not None
        assert hasattr(pde, "mse")
        assert hasattr(pde, "volume_grad_expr")

    def test_scalar_nonlinear_volume_weak_form_assembles(self):
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = make_scalar_net()(xi, yi)
        weak = (1.0 + u_net**2) * (jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi)) - (
            1.0 + 0.0 * xi
        ) * vi

        pde = jno.fem([weak, u(xb, yb) - 0.0])

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Boundary-tagged VPINN assembly
# ============================================================


class TestVpinnBoundaryAssembly:
    @pytest.mark.xfail(
        reason=(
            "jno.fem VPINN does not yet lower a Neumann *flux* boundary term: a bound-test boundary "
            "value term's region is not propagated into the VPINN channel bucketing. The FEM path "
            "classifies it via _region_and_support, but the VPINN path buckets on Variable.fem_meta, "
            "which a bound test does not carry -- so the flux is filed under the volume channel. This "
            "is a pre-existing jno.fem VPINN-lowering gap, orthogonal to the native fem_context."
        ),
        strict=True,
    )
    def test_volume_plus_boundary_weak_form_assembles(self):
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        xr, yr, _ = dom.variable("right", split=True)
        vi = phi.bind(x=xi, y=yi)
        vr = phi.bind(x=xr, y=yr)
        u_net = make_scalar_net()(xi, yi)

        vol = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi)
        surf = (1.0 + 0.0 * xr) * vr  # a Neumann (boundary test) flux term on 'right'
        pde = jno.fem([vol, surf, u(xb, yb) - 0.0])

        assert hasattr(pde, "mse")
        assert "right" in pde.boundary_value_exprs  # <-- the gap: lands in the volume channel instead


# ============================================================
# Vector VPINN assembly
# ============================================================


class TestVpinnVectorAssembly:
    def test_vector_weak_form_assembles(self):
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = make_vector_net()(xi, yi)

        eps_u = jnn.symgrad(u_net, [xi, yi])
        eps_phi = jnn.symgrad(vi, [xi, yi])
        weak = jnn.inner(eps_u, eps_phi, n_contract=2)

        pde = jno.fem([weak, u(xb, yb) - (0.0, 0.0)])

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Validation / display
# ============================================================


class TestVpinnValidation:
    def test_invalid_tag_raises(self):
        dom = make_domain()

        with pytest.raises(Exception):
            dom.variable("does_not_exist", split=True)


def test_dump_tree_on_vpinn_weak_form():
    dom = make_domain()
    init_vpinn_fem(dom, with_neumann_tags=False)

    u, phi = dom.fem_symbols()
    x, y, _ = dom.variable("fem_gauss", split=True)

    weak = jnn.grad(u, x) * jnn.grad(phi, x) + 0.0 * y
    tree = dump_tree(weak)

    assert isinstance(tree, str)
    assert len(tree) > 0


def test_vpinn_via_jno_fem_solves_poisson():
    """VPINN entirely through dom.fem_symbols() + jno.fem (no init_fem, no weak.assemble): a network
    trial u=net(x,y) written into the weak form is detected (ModelCall) and test-projected onto the
    FE test space; the Dirichlet condition u(boundary)-0 declares which test functions vanish on the
    boundary (so their du/dn-flux residual is masked). Trains to the analytic Poisson solution."""
    import numpy as np

    optax = pytest.importorskip("optax")

    dom = make_domain(mesh_size=0.2)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("boundary", split=True)
    net = jnn.nn.wrap(foundax.mlp(2, hidden_dims=24, num_layers=3, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
    bc = xi * (1 - xi) * yi * (1 - yi)
    u_net = net(xi, yi) * bc  # hard-BC ansatz: vanishes on the [0,1]^2 boundary
    phii = phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))  # -lap(x(1-x)y(1-y)) = f

    weak = jnn.grad(u_net, xi) * jnn.grad(phii, xi) + jnn.grad(u_net, yi) * jnn.grad(phii, yi) - f * phii
    pde = jno.fem([weak, u(xb, yb) - 0.0])  # net trial + Dirichlet (masks boundary test functions)
    assert type(pde).__name__ == "GroupedAssembly" and hasattr(pde, "mse")

    net.optimizer(optax.adam(1e-2))
    crux = jno.core([pde.mse], domain=dom)
    crux.solve(1500)

    # verify the trained net solves the PDE on a fresh grid (eval the prediction, not the loss)
    test_dom = make_domain(mesh_size=0.12)
    xt, yt, _ = test_dom.variable("interior", split=True)
    bc_t = xt * (1 - xt) * yt * (1 - yt)
    pred = np.asarray(crux.eval([net(xt, yt) * bc_t], domain=test_dom)).reshape(-1)
    exact = np.asarray(crux.eval([bc_t], domain=test_dom)).reshape(-1)
    rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
    assert rel < 1e-2, f"VPINN did not solve Poisson: rel-L2={rel:.3e}"


@pytest.mark.parametrize("opt_name", ["adam", "sgd", "adamw", "rmsprop"])
def test_network_trains_under_x64_with_optax_optimizer(opt_name):
    """Network training under jax_enable_x64 (float64 params) works for ANY optax optimizer: jNO
    casts the optimizer state to the param precision, so optax's float32-default moment/LR state does
    not clash with the float64 params (the optimizer-state dtype mismatch this guards against)."""
    import numpy as np
    import optax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        opt = {
            "adam": optax.adam(1e-2),
            "sgd": optax.sgd(5e-2),
            "adamw": optax.adamw(1e-2),
            "rmsprop": optax.rmsprop(2e-3),
        }[opt_name]
        dom = make_domain(mesh_size=0.35)
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        net = make_scalar_net()
        bc = xi * (1 - xi) * yi * (1 - yi)
        u_net = net(xi, yi) * bc
        vi = phi.bind(x=xi, y=yi)
        f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
        weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - f * vi
        pde = jno.fem([weak, u(xb, yb) - 0.0])
        net.optimizer(opt)
        crux = jno.core([pde.mse], domain=dom)
        l0 = float(np.asarray(crux.eval([pde.mse])).mean())
        crux.solve(300)
        l1 = float(np.asarray(crux.eval([pde.mse])).mean())
        assert l1 < 0.5 * l0, f"{opt_name} under x64 did not train: {l0:.2e} -> {l1:.2e}"
    finally:
        jax.config.update("jax_enable_x64", prev)
