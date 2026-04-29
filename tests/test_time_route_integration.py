"""Lightweight tests for time-dependent FEAX-time and Diffrax routing."""

import pytest

pytest.importorskip("feax", reason="feax required for FEAX-time route tests")
pytest.importorskip("diffrax", reason="diffrax required for Diffrax adapter tests")

import numpy as np
import jax.numpy as jnp

import jno
import jno.jnp_ops as jnn
from jno.utils.solver.backend_blocks import DiffraxBlock, FeaxPipelineBlock, FeaxTimeBlock


# ============================================================
# Helpers
# ============================================================


def make_time_domain(mesh_size=0.45, time=(0.0, 0.05, 3)):
    """Create a tiny time-dependent domain for route tests."""
    return jno.domain(
        constructor=jno.domain.rect(mesh_size=mesh_size),
        time=time,
        compute_mesh_connectivity=False,
    )


def init_time_fem(dom):
    """Initialize a minimal scalar FEAX/FEM context for transient tests."""
    dom.init_fem(
        element_type="TRI3",
        quad_degree=2,
        bcs=[dom.dirichlet(["left", "right", "bottom", "top"], 0.0)],
        fem_solver=True,
    )
    return dom


def zero_state_from_mesh(dom):
    """Return a zero nodal state matching the scalar FEM mesh."""
    n_nodes = int(np.asarray(dom.mesh.points).shape[0])
    return jnp.zeros((n_nodes,), dtype=jnp.float32)


def make_heat_weak_form(dom):
    """
    Build a tiny linear heat-equation weak form:

        ∫ u_t phi + nu ∫ grad(u) · grad(phi) = 0
    """
    nu = 0.1

    u, phi = dom.fem_symbols()
    xg, yg, tg = dom.variable("fem_gauss", split=True)

    u_t = jnn.grad(u, tg)
    ux = jnn.grad(u, xg)
    uy = jnn.grad(u, yg)

    phix = jnn.grad(phi, xg)
    phiy = jnn.grad(phi, yg)

    weak = u_t * phi + nu * (ux * phix + uy * phiy)
    return weak


def make_nonlinear_reaction_weak_form(dom):
    """
    Build a tiny nonlinear transient weak form:

        ∫ u_t phi + ∫ grad(u) · grad(phi) + ∫ (u^3 - u) phi = 0
    """
    u, phi = dom.fem_symbols()
    xg, yg, tg = dom.variable("fem_gauss", split=True)

    u_t = jnn.grad(u, tg)
    ux = jnn.grad(u, xg)
    uy = jnn.grad(u, yg)

    phix = jnn.grad(phi, xg)
    phiy = jnn.grad(phi, yg)

    weak = (
        u_t * phi
        + (ux * phix + uy * phiy)
        + (u * u * u - u) * phi
    )
    return weak


# ============================================================
# Linear FEAX-time route
# ============================================================


class TestLinearFeaxTimeRoute:
    def test_linear_weak_form_assembles_to_feax_time_block(self):
        dom = init_time_fem(make_time_domain())
        weak = make_heat_weak_form(dom)
        state0 = zero_state_from_mesh(dom)

        block = weak.assemble(
            dom,
            target="feax_time",
            linear=True,
            state0=state0,
            mode="implicit",
        )

        assert isinstance(block, FeaxTimeBlock)
        assert block.backend == "feax_time"
        assert block.time_order == 1
        assert block.is_linear()
        assert not block.is_nonlinear()

        assert block.M is not None
        assert block.A is not None
        assert block.state0 is not None
        assert block.feax_mesh is not None

        M = jnp.asarray(block.M)
        A = jnp.asarray(block.A)

        assert M.ndim == 2
        assert A.ndim == 2
        assert M.shape[0] == M.shape[1]
        assert A.shape[0] == A.shape[1]
        assert M.shape == A.shape
        assert M.shape[0] == state0.shape[0]

    def test_linear_feax_time_block_converts_to_diffrax_block(self):
        dom = init_time_fem(make_time_domain())
        weak = make_heat_weak_form(dom)
        state0 = zero_state_from_mesh(dom)

        block = weak.assemble(
            dom,
            target="feax_time",
            linear=True,
            state0=state0,
            mode="explicit",
        )

        dblock = block.as_diffrax()

        assert isinstance(dblock, DiffraxBlock)
        assert dblock.backend == "diffrax"
        assert dblock.time_order == 1
        assert dblock.rhs is not None
        assert dblock.term is not None
        assert dblock.state0 is block.state0 or dblock.state0 is not None

        rhs0 = dblock.rhs(dblock.t0, dblock.state0, None)
        rhs0 = jnp.asarray(rhs0)

        assert rhs0.shape == state0.shape
        assert jnp.isfinite(rhs0).all()

    def test_linear_feax_time_block_converts_to_feax_pipeline_block(self):
        dom = init_time_fem(make_time_domain())
        weak = make_heat_weak_form(dom)
        state0 = zero_state_from_mesh(dom)

        block = weak.assemble(
            dom,
            target="feax_time",
            linear=True,
            state0=state0,
            mode="implicit",
        )

        pblock = block.as_feax_pipeline(
            scheme="backward_euler",
            compile_step=False,
        )

        assert isinstance(pblock, FeaxPipelineBlock)
        assert pblock.backend == "feax_time"
        assert pblock.scheme == "backward_euler"
        assert pblock.pipeline is not None
        assert pblock.mesh is not None
        assert pblock.state0 is not None

        cfg = pblock.make_time_config(print_every=1, save_every=1)
        assert cfg is not None


# ============================================================
# Nonlinear FEAX-time route
# ============================================================


class TestNonlinearFeaxTimeRoute:
    def test_nonlinear_weak_form_assembles_to_feax_time_block(self):
        dom = init_time_fem(make_time_domain())
        weak = make_nonlinear_reaction_weak_form(dom)
        state0 = zero_state_from_mesh(dom)

        block = weak.assemble(
            dom,
            target="feax_time",
            state0=state0,
            mode="implicit",
        )

        assert isinstance(block, FeaxTimeBlock)
        assert block.backend == "feax_time"
        assert block.time_order == 1
        assert block.is_nonlinear()
        assert not block.is_linear()

        assert block.mass is not None
        assert block.residual is not None
        assert block.jacobian is not None
        assert block.state0 is not None
        assert block.feax_mesh is not None

        M0 = jnp.asarray(block.mass(block.t0, None))
        R0 = jnp.asarray(block.residual(block.state0, block.t0, None))
        J0 = jnp.asarray(block.jacobian(block.state0, block.t0, None))

        assert M0.ndim == 2
        assert M0.shape[0] == M0.shape[1]
        assert R0.shape == state0.shape
        assert J0.shape == (state0.shape[0], state0.shape[0])

        assert jnp.isfinite(M0).all()
        assert jnp.isfinite(R0).all()
        assert jnp.isfinite(J0).all()

    def test_nonlinear_feax_time_block_converts_to_diffrax_block(self):
        dom = init_time_fem(make_time_domain())
        weak = make_nonlinear_reaction_weak_form(dom)
        state0 = zero_state_from_mesh(dom)

        block = weak.assemble(
            dom,
            target="feax_time",
            state0=state0,
            mode="implicit",
        )

        dblock = block.as_diffrax()

        assert isinstance(dblock, DiffraxBlock)
        assert dblock.backend == "diffrax"
        assert dblock.form == "explicit_first_order_nonlinear"
        assert dblock.rhs is not None
        assert dblock.term is not None

        rhs0 = jnp.asarray(dblock.rhs(dblock.t0, dblock.state0, None))

        assert rhs0.shape == state0.shape
        assert jnp.isfinite(rhs0).all()

    def test_nonlinear_feax_time_block_converts_to_feax_pipeline_block(self):
        dom = init_time_fem(make_time_domain())
        weak = make_nonlinear_reaction_weak_form(dom)
        state0 = zero_state_from_mesh(dom)

        block = weak.assemble(
            dom,
            target="feax_time",
            state0=state0,
            mode="implicit",
        )

        pblock = block.as_feax_pipeline(
            scheme="backward_euler",
            newton_maxiter=2,
            compile_step=False,
        )

        assert isinstance(pblock, FeaxPipelineBlock)
        assert pblock.backend == "feax_time"
        assert pblock.scheme == "backward_euler"
        assert pblock.pipeline is not None
        assert pblock.mesh is not None

        cfg = pblock.make_time_config(print_every=1, save_every=1)
        assert cfg is not None


# ============================================================
# Validation behavior
# ============================================================


class TestTimeRouteValidation:
    def test_feax_time_requires_temporal_derivative(self):
        dom = init_time_fem(make_time_domain())

        u, phi = dom.fem_symbols()
        xg, yg, _ = dom.variable("fem_gauss", split=True)

        weak = jnn.grad(u, xg) * jnn.grad(phi, xg) + 0.0 * yg

        with pytest.raises(ValueError):
            weak.assemble(
                dom,
                target="feax_time",
                state0=zero_state_from_mesh(dom),
            )