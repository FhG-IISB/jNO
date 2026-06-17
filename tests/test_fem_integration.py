"""Lightweight tests for FEAX-backed FEM / weak-form integration."""

import pytest

pytest.importorskip("feax", reason="feax required for FEAX FEM integration tests")

import jax.numpy as jnp

import jno
import jno.jnp_ops as jnn
from jno import dirichlet

# ============================================================
# Helpers
# ============================================================


def make_domain(mesh_size=0.35):
    """Create a small rectangular domain for fast FEAX/FEM tests."""
    return jno.domain(constructor=jno.domain.rect(mesh_size=mesh_size))


def to_dense(A):
    """Convert sparse/JAX/scipy-like matrices to a dense JAX array."""
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# ============================================================
# FEM init / context
# ============================================================


class TestFeaxFemInit:
    def test_init_fem_registers_volume_and_boundary_quadrature(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[
                dom.dirichlet("left"),
                dom.neumann(["right", "top"]),
            ],
            fem_solver=True,
        )

        xg, yg, _ = dom.variable("fem_gauss", split=True)
        xr, yr, _ = dom.variable("gauss_right", split=True)
        xt, yt, _ = dom.variable("gauss_top", split=True)

        assert xg is not None
        assert yg is not None
        assert xr is not None
        assert yr is not None
        assert xt is not None
        assert yt is not None

        assert "fem_gauss" in dom._mesh_pool
        assert "gauss_right" in dom._mesh_pool
        assert "gauss_top" in dom._mesh_pool

        assert dom._mesh_pool["fem_gauss"].shape[-1] == dom.dimension
        assert dom._mesh_pool["gauss_right"].shape[-1] == dom.dimension
        assert dom._mesh_pool["gauss_top"].shape[-1] == dom.dimension

    def test_default_zero_dirichlet_builds(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[dirichlet(["left", "bottom"])],
            fem_solver=True,
        )

        u, phi = dom.fem_symbols()
        xg, yg, _ = dom.variable("fem_gauss", split=True)

        assert u is not None
        assert phi is not None
        assert xg is not None
        assert yg is not None
        assert "fem_gauss" in dom._mesh_pool


# ============================================================
# FEM symbols
# ============================================================


class TestFeaxFemSymbols:
    def test_scalar_fem_symbols(self):
        dom = make_domain()
        u, phi = dom.fem_symbols()

        assert getattr(u, "value_shape", ()) == ()
        assert getattr(phi, "value_shape", ()) == ()

    def test_vector_fem_symbols(self):
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))

        assert getattr(u, "value_shape", None) == (2,)
        assert getattr(phi, "value_shape", None) == (2,)


# ============================================================
# Linear scalar FEM assembly
# ============================================================


class TestFeaxFemLinearAssembly:
    def test_scalar_poisson_fem_system_assembles(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[dom.dirichlet(["left", "right", "top", "bottom"])],
            fem_solver=True,
        )

        u, phi = dom.fem_symbols()
        xg, yg, _ = dom.variable("fem_gauss", split=True)

        ux = jnn.grad(u, xg)
        uy = jnn.grad(u, yg)
        phix = jnn.grad(phi, xg)
        phiy = jnn.grad(phi, yg)

        weak = ux * phix + uy * phiy - (1.0 + 0.0 * xg) * phi
        A, b = weak.assemble(dom, target="fem_system")

        A_dense = to_dense(A)
        b = jnp.asarray(b)

        assert A_dense.ndim == 2
        assert b.ndim == 1
        assert A_dense.shape[0] == A_dense.shape[1]
        assert A_dense.shape[0] == b.shape[0]
        assert jnp.isfinite(A_dense).all()
        assert jnp.isfinite(b).all()

    def test_target_none_auto_selects_fem_system_for_linear_steady_form(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[dom.dirichlet(["left", "right", "top", "bottom"])],
            fem_solver=True,
        )

        u, phi = dom.fem_symbols()
        xg, yg, _ = dom.variable("fem_gauss", split=True)

        weak = jnn.grad(u, xg) * jnn.grad(phi, xg) - (1.0 + 0.0 * xg) * phi

        A, b = weak.assemble(dom)

        A_dense = to_dense(A)
        b = jnp.asarray(b)

        assert A_dense.shape[0] == A_dense.shape[1]
        assert A_dense.shape[0] == b.shape[0]


# ============================================================
# Linear vector FEM assembly
# ============================================================


class TestFeaxFemVectorAssembly:
    def test_linear_elasticity_like_system_assembles(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[dom.dirichlet(["left", "right", "top", "bottom"], (0.0, 0.0))],
            fem_solver=True,
            vec=2,
        )

        u, phi = dom.fem_symbols(value_shape=(2,))
        xg, yg, _ = dom.variable("fem_gauss", split=True)

        eps_u = jnn.symgrad(u, [xg, yg])
        eps_phi = jnn.symgrad(phi, [xg, yg])

        weak = jnn.inner(eps_u, eps_phi, n_contract=2)
        A, b = weak.assemble(dom, target="fem_system")

        A_dense = to_dense(A)
        b = jnp.asarray(b)

        assert A_dense.ndim == 2
        assert b.ndim == 1
        assert A_dense.shape[0] == A_dense.shape[1]
        assert A_dense.shape[0] == b.shape[0]
        assert jnp.isfinite(A_dense).all()
        assert jnp.isfinite(b).all()

    def test_full_elasticity_volumetric_term_recovers_manufactured(self):
        # Full lambda-mu linear elasticity *including* the volumetric div(u)*div(phi)
        # term (trace(symgrad)*trace(symgrad)) — this requires the kernel's
        # prefix-alignment of trial- vs test-derived quantities. u_exact = (a x, b y)
        # has constant strain so div(sigma) = 0 (f = 0); clamping it on the whole
        # boundary must recover it exactly on TRI3 (linear field).
        import numpy as np

        lam, mu, a, bcoef = 1.0, 1.0, 0.1, -0.05
        dom = make_domain()
        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[dom.dirichlet(["left", "right", "top", "bottom"], [lambda p: a * p[0], lambda p: bcoef * p[1]])],
            fem_solver=True,
            vec=2,
        )
        u, phi = dom.fem_symbols(value_shape=(2,))
        xg, yg, _ = dom.variable("fem_gauss", split=True)
        eps_u, eps_phi = jnn.symgrad(u, [xg, yg]), jnn.symgrad(phi, [xg, yg])
        weak = lam * jnn.trace(eps_u) * jnn.trace(eps_phi) + 2.0 * mu * jnn.inner(eps_u, eps_phi, n_contract=2)
        A, rhs = weak.assemble(dom, target="fem_system")
        sol = np.linalg.solve(to_dense(A), np.asarray(rhs).reshape(-1))
        c = np.asarray(dom.mesh.points)[:, :2]
        exact = np.stack([a * c[:, 0], bcoef * c[:, 1]], axis=-1).reshape(-1)
        rel = np.linalg.norm(exact - sol) / (np.linalg.norm(exact) + 1e-30)
        assert rel < 1e-8


# ============================================================
# Boundary-tagged weak forms
# ============================================================


class TestFeaxFemBoundaryAssembly:
    def test_neumann_boundary_quadrature_tag_is_usable(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[
                dom.dirichlet("left"),
                dom.neumann("right"),
            ],
            fem_solver=True,
        )

        u, phi = dom.fem_symbols()
        xg, yg, _ = dom.variable("fem_gauss", split=True)
        xr, yr, _ = dom.variable("gauss_right", split=True)

        ux = jnn.grad(u, xg)
        uy = jnn.grad(u, yg)
        phix = jnn.grad(phi, xg)
        phiy = jnn.grad(phi, yg)

        vol = ux * phix + uy * phiy
        surf = (1.0 + 0.0 * xr + 0.0 * yr) * phi
        weak = vol - surf

        A, b = weak.assemble(dom, target="fem_system")

        A_dense = to_dense(A)
        b = jnp.asarray(b)

        assert A_dense.shape[0] == A_dense.shape[1]
        assert A_dense.shape[0] == b.shape[0]
        assert jnp.isfinite(A_dense).all()
        assert jnp.isfinite(b).all()


# ============================================================
# Nonlinear FEM residual route
# ============================================================


class TestFeaxFemResidualRoute:
    def test_nonlinear_residual_operator_builds(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[dom.dirichlet(["left", "right", "top", "bottom"])],
            fem_solver=True,
        )

        u, phi = dom.fem_symbols()
        xg, yg, _ = dom.variable("fem_gauss", split=True)

        ux = jnn.grad(u, xg)
        uy = jnn.grad(u, yg)
        phix = jnn.grad(phi, xg)
        phiy = jnn.grad(phi, yg)

        weak = (1.0 + u**2) * (ux * phix + uy * phiy) - (1.0 + 0.0 * xg) * phi
        op = weak.assemble(dom, target="fem_residual")

        assert hasattr(op, "size")
        assert hasattr(op, "residual")
        assert hasattr(op, "jacobian")
        assert op.size > 0

        u0 = jnp.zeros(op.size)
        r0 = op.residual(u0)
        J0 = op.jacobian(u0)

        J0_dense = to_dense(J0)

        assert r0.shape == (op.size,)
        assert J0_dense.shape == (op.size, op.size)
        assert jnp.isfinite(r0).all()
        assert jnp.isfinite(J0_dense).all()


# ============================================================
# BC validation / registration
# ============================================================


class TestFeaxFemBCNormalization:
    def test_vector_dirichlet_component_dict_builds(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            vec=2,
            bcs=[
                dom.dirichlet("left", {"x": 0.0}),
                dom.dirichlet("bottom", {"y": 0.0}),
            ],
            fem_solver=True,
        )

        assert "left" in dom._fem_dirichlet_tags
        assert "bottom" in dom._fem_dirichlet_tags
        assert dom._fem_dirichlet_value_fns is not None
        assert "left" in dom._fem_dirichlet_value_fns
        assert "bottom" in dom._fem_dirichlet_value_fns


class TestFeaxFemBCValidation:
    def test_dirichlet_tuple_length_mismatch_raises(self):
        dom = make_domain()

        with pytest.raises(ValueError):
            dom.init_fem(
                element_type="TRI3",
                quad_degree=2,
                vec=2,
                bcs=[dom.dirichlet("left", (0.0, 0.0, 0.0))],
                fem_solver=True,
            )

    def test_mixing_bcs_and_legacy_args_raises(self):
        dom = make_domain()

        with pytest.raises(ValueError):
            dom.init_fem(
                element_type="TRI3",
                quad_degree=2,
                dirichlet_tags=["left"],
                bcs=[dirichlet("right")],
                fem_solver=True,
            )


class TestFeaxFemSurfaceRegistration:
    def test_multiple_neumann_tags_register_surface_data(self):
        dom = make_domain()

        dom.init_fem(
            element_type="TRI3",
            quad_degree=2,
            bcs=[
                dom.dirichlet("left"),
                dom.neumann(["right", "top"]),
            ],
            fem_solver=True,
        )

        assert "surface_data" in dom.fem_context
        assert "right" in dom.fem_context["surface_data"]
        assert "top" in dom.fem_context["surface_data"]

        right_data = dom.fem_context["surface_data"]["right"]
        top_data = dom.fem_context["surface_data"]["top"]

        assert "quad_points" in right_data
        assert "quad_points" in top_data

        assert right_data["quad_points"].shape[-1] == dom.dimension
        assert top_data["quad_points"].shape[-1] == dom.dimension
