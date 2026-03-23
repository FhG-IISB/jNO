"""Minimal tests for the VPINN / weak-form integration."""

import pytest
import jax
import jax.numpy as jnp
import jno
import jno.numpy as jnn
from jno.trace import dump_tree


# ============================================================
# Helpers
# ============================================================


def make_domain(mesh_size=0.25):
    """Create a small rectangular domain for fast VPINN tests."""
    return jno.domain(constructor=jno.domain.rect(mesh_size=mesh_size))


def init_vpinn_fem(dom, with_neumann_tags=True):
    """
    Initialize FEM quadrature tags used by the VPINN route.

    VPINN uses the same sampled tags as the FEM route:
      - fem_gauss      : volume quadrature
      - gauss_<tag>    : boundary quadrature
    """
    bcs = [dom.dirichlet("left")]
    if with_neumann_tags:
        bcs.append(dom.neumann(["right", "top"]))

    dom.init_fem(
        element_type="TRI3",
        quad_degree=2,
        bcs=bcs,
        fem_solver=True,
    )
    return dom


def make_scalar_net():
    key = jax.random.PRNGKey(0)
    return jnn.nn.mlp(
        2,
        hidden_dims=32,
        num_layers=2,
        activation=jax.nn.tanh,
        key=key,
    )


def make_vector_net():
    key = jax.random.PRNGKey(0)
    return jnn.nn.mlp(
        2,
        hidden_dims=32,
        num_layers=2,
        activation=jax.nn.tanh,
        key=key,
        output_dim=2,
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
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=False)

        u, phi = dom.fem_symbols()
        x, y, _ = dom.variable("fem_gauss", split=True)

        ux = jnn.grad(u, x)
        uy = jnn.grad(u, y)
        phix = jnn.grad(phi, x)
        phiy = jnn.grad(phi, y)

        weak = ux * phix + uy * phiy - (1.0 + 0.0 * x) * phi

        net = make_scalar_net()
        u_net = net(x, y)

        pde = weak.assemble(dom, u_net=u_net, target="vpinn")

        assert pde is not None
        assert hasattr(pde, "mse")

    def test_scalar_nonlinear_volume_weak_form_assembles(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=False)

        u, phi = dom.fem_symbols()
        x, y, _ = dom.variable("fem_gauss", split=True)

        ux = jnn.grad(u, x)
        uy = jnn.grad(u, y)
        phix = jnn.grad(phi, x)
        phiy = jnn.grad(phi, y)

        weak = (1.0 + u**2) * (ux * phix + uy * phiy) - (1.0 + 0.0 * x) * phi

        net = make_scalar_net()
        u_net = net(x, y)

        pde = weak.assemble(dom, u_net=u_net, target="vpinn")

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Boundary-tagged VPINN assembly
# ============================================================


class TestVpinnBoundaryAssembly:
    def test_volume_plus_boundary_weak_form_assembles(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=True)

        u, phi = dom.fem_symbols()

        x, y, _ = dom.variable("fem_gauss", split=True)
        xr, yr, _ = dom.variable("gauss_right", split=True)

        ux = jnn.grad(u, x)
        uy = jnn.grad(u, y)
        phix = jnn.grad(phi, x)
        phiy = jnn.grad(phi, y)

        vol = ux * phix + uy * phiy
        surf = (1.0 + 0.0 * xr) * phi
        weak = vol - surf

        net = make_scalar_net()
        u_net = net(x, y)

        pde = weak.assemble(dom, u_net=u_net, target="vpinn")

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Vector VPINN assembly
# ============================================================


class TestVpinnVectorAssembly:
    def test_vector_weak_form_assembles(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=False)

        u, phi = dom.fem_symbols(value_shape=(2,))
        x, y, _ = dom.variable("fem_gauss", split=True)

        eps_u = jnn.symgrad(u, [x, y])
        eps_phi = jnn.symgrad(phi, [x, y])

        weak = jnn.inner(eps_u, eps_phi, n_contract=2)

        net = make_vector_net()
        u_net = net(x, y)

        pde = weak.assemble(dom, u_net=u_net, target="vpinn")

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Validation / error behavior
# ============================================================


class TestVpinnValidation:
    def test_invalid_tag_raises(self):
        dom = make_domain()

        with pytest.raises(Exception):
            dom.variable("does_not_exist", split=True)


# ============================================================
# Symbolic weak-form display
# ============================================================


def test_dump_tree_on_vpinn_weak_form():
    dom = make_domain()
    init_vpinn_fem(dom, with_neumann_tags=False)

    u, phi = dom.fem_symbols()
    x, y, _ = dom.variable("fem_gauss", split=True)

    weak = jnn.grad(u, x) * jnn.grad(phi, x) + 0.0 * y
    tree = dump_tree(weak)

    assert isinstance(tree, str)
    assert len(tree) > 0
