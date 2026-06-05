"""Lightweight tests for periodic BCs and affine / non-affine runtime parameters.

These lock in the behaviour verified by hand in check_periodic_bc.py:

  * periodic reduction builds a prolongation P and a reduced (n_red < n_full)
    semidiscrete system, on a structured (periodic-compatible) mesh;
  * an AFFINE runtime parameter (nu * grad u . grad phi) is exposed through the
    operator and reproduces the analytical periodic-heat decay on both the
    Diffrax and FEAX-pipeline routes;
  * a NON-AFFINE runtime parameter (exp(logk) * grad u . grad phi, parameter
    inside a nonlinear function) routes through the native non-affine operator
    path, is reported in the block metadata, and -- because exp(log 0.1) = 0.1 --
    reproduces the SAME analytical solution as the affine case.

The analytical reference is the separable periodic heat mode

    u(x,y,t) = exp(-8 pi^2 nu t) sin(2 pi x) sin(2 pi y),   nu = 0.1.
"""

import pytest

pytest.importorskip("feax", reason="feax required for periodic / parametric route tests")
pytest.importorskip("diffrax", reason="diffrax required for the Diffrax route test")

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import jno
import jno.jnp_ops as jnn

jax.config.update("jax_enable_x64", False)

# ============================================================
# Shared constants
# ============================================================

NU = 0.1
T_END = 0.2
N_TIME = 11
N_GRID = 12
TWO_PI = 2.0 * np.pi
PASS_TOL = 0.05  # P1 + backward-Euler discretisation error is a few %


# ============================================================
# Helpers
# ============================================================


def make_periodic_domain():
    """Structured (periodic-compatible) rectangular domain on [0,1]^2."""
    return jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=N_GRID, ny=N_GRID),
        time=(0.0, T_END, N_TIME),
        compute_mesh_connectivity=False,
    )


def init_periodic_fem(dom):
    dom.init_fem(
        element_type="TRI3",
        quad_degree=3,
        bcs=[dom.periodic(("left", "right"), ("bottom", "top"))],
        fem_solver=True,
    )
    return dom


def periodic_state0(dom):
    xy = np.asarray(dom.mesh.points)[:, :2]
    u0 = np.sin(TWO_PI * xy[:, 0]) * np.sin(TWO_PI * xy[:, 1])
    return jnp.asarray(u0, dtype=jnp.float32)


def build_periodic_heat_block(dom, kind):
    """kind in {'affine', 'nonaffine'}; returns (block, args)."""
    u, phi = dom.fem_symbols()
    xg, yg, tg = dom.variable("fem_gauss", split=True)
    u_t = jnn.grad(u, tg)
    ux = jnn.grad(u, xg)
    uy = jnn.grad(u, yg)
    phix = jnn.grad(phi, xg)
    phiy = jnn.grad(phi, yg)
    diffusion = ux * phix + uy * phiy

    if kind == "affine":
        nu = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="nu")
        nu.initialize(jax.nn.initializers.constant(NU))
        weak = u_t * phi + nu * diffusion
        args = {"nu": NU}
    elif kind == "nonaffine":
        logk = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="logk")
        logk.initialize(jax.nn.initializers.constant(float(np.log(NU))))
        weak = u_t * phi + jno.np.exp(logk) * diffusion  # parameter inside exp
        args = {"logk": float(np.log(NU))}
    else:
        raise ValueError(kind)

    block = weak.assemble(
        dom,
        target="fem_time",
        linear=True,
        state0=periodic_state0(dom),
        mode="implicit",
    )
    return block, args


def analytical(mesh_xy, save_ts):
    spatial = np.sin(TWO_PI * mesh_xy[:, 0]) * np.sin(TWO_PI * mesh_xy[:, 1])
    decay = np.exp(-8.0 * np.pi**2 * NU * np.asarray(save_ts))[:, None]
    return decay * spatial[None, :]


def relative_l2(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def prolongation(dom):
    return jnp.asarray(dom._feax_context["P"])


def solve_diffrax(dom, block, args, save_ts):
    dblock = block.as_diffrax()
    P = prolongation(dom)
    sol = diffrax.diffeqsolve(
        dblock.term,
        diffrax.Tsit5(),
        t0=0.0,
        t1=T_END,
        dt0=1e-3,
        y0=jnp.asarray(dblock.state0),
        args=args,
        saveat=diffrax.SaveAt(ts=jnp.asarray(save_ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        max_steps=200_000,
    )
    return np.asarray(jnp.asarray(sol.ys) @ P.T)


def solve_feax_pipeline(dom, block, args, save_ts, sub=8):
    pblock = block.as_feax_pipeline(scheme="backward_euler", args=args)
    pipe = pblock.pipeline
    pipe.build(pblock.mesh)
    P = prolongation(dom)
    state = jnp.asarray(pipe.initial_state())
    save_ts = np.asarray(save_ts)
    dt = float(save_ts[1] - save_ts[0]) / sub
    states = [state]
    t = 0.0
    for _ in range(1, len(save_ts)):
        for _ in range(sub):
            state = pipe.step(state, t, dt)
            t += dt
        states.append(state)
    return np.asarray(jnp.stack(states) @ P.T)


# ============================================================
# Periodic reduction structure
# ============================================================


class TestPeriodicReductionStructure:
    def test_init_fem_builds_prolongation_and_reduces_dofs(self):
        dom = init_periodic_fem(make_periodic_domain())

        assert "P" in dom._feax_context
        assert "periodic" in dom._feax_context

        P = np.asarray(dom._feax_context["P"])
        info = dom._feax_context["periodic"]
        n_full = int(np.asarray(dom.mesh.points).shape[0])

        assert P.ndim == 2
        assert P.shape[0] == n_full  # full rows
        assert P.shape[1] < n_full  # reduced columns
        assert info["n_full"] == n_full
        assert info["n_red"] == P.shape[1]
        # Each row of P selects exactly one master DOF (partition-of-unity rows).
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_periodic_block_carries_reduced_system_and_prolongation(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, _ = build_periodic_heat_block(dom, "affine")

        n_full = int(np.asarray(dom.mesh.points).shape[0])
        assert bool(block.metadata.get("periodic")) is True
        assert block.metadata["full_state_size"] == n_full
        assert block.metadata["reduced_state_size"] < n_full

        M = jnp.asarray(block.M)
        assert M.shape[0] == M.shape[1]
        assert M.shape[0] == block.metadata["reduced_state_size"]
        assert getattr(block, "prolongation", None) is not None


# ============================================================
# Affine runtime parameter
# ============================================================


class TestPeriodicAffineParameter:
    def test_affine_parameter_reported(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, _ = build_periodic_heat_block(dom, "affine")

        assert block.operator_fn is not None
        assert "nu" in list(block.metadata.get("runtime_parameter_names", []))

    def test_affine_periodic_matches_analytical_diffrax(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, args = build_periodic_heat_block(dom, "affine")
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        mesh_xy = np.asarray(dom.mesh.points)[:, :2]
        err = relative_l2(solve_diffrax(dom, block, args, save_ts), analytical(mesh_xy, save_ts))
        assert err < PASS_TOL, f"affine/diffrax relL2={err:.3e}"

    def test_affine_periodic_matches_analytical_feax_pipeline(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, args = build_periodic_heat_block(dom, "affine")
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        mesh_xy = np.asarray(dom.mesh.points)[:, :2]
        err = relative_l2(solve_feax_pipeline(dom, block, args, save_ts), analytical(mesh_xy, save_ts))
        assert err < PASS_TOL, f"affine/feax-pipeline relL2={err:.3e}"


# ============================================================
# Non-affine runtime parameter
# ============================================================


class TestPeriodicNonAffineParameter:
    def test_nonaffine_parameter_routed_and_reported(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, _ = build_periodic_heat_block(dom, "nonaffine")

        assert block.operator_fn is not None
        assert bool(block.metadata.get("nonaffine_operator")) is True
        assert "logk" in list(block.metadata.get("runtime_parameter_names", []))

    def test_nonaffine_operator_is_nonzero_at_args(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, args = build_periodic_heat_block(dom, "nonaffine")

        A = jnp.asarray(block.operator_fn(0.0, {k: jnp.asarray(v) for k, v in args.items()}))
        n_red = block.metadata["reduced_state_size"]
        assert A.shape == (n_red, n_red)  # operator is reduced
        assert float(jnp.linalg.norm(A)) > 0.0

    def test_nonaffine_periodic_matches_analytical_diffrax(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, args = build_periodic_heat_block(dom, "nonaffine")
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        mesh_xy = np.asarray(dom.mesh.points)[:, :2]
        err = relative_l2(solve_diffrax(dom, block, args, save_ts), analytical(mesh_xy, save_ts))
        assert err < PASS_TOL, f"nonaffine/diffrax relL2={err:.3e}"

    def test_nonaffine_periodic_matches_analytical_feax_pipeline(self):
        dom = init_periodic_fem(make_periodic_domain())
        block, args = build_periodic_heat_block(dom, "nonaffine")
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        mesh_xy = np.asarray(dom.mesh.points)[:, :2]
        err = relative_l2(solve_feax_pipeline(dom, block, args, save_ts), analytical(mesh_xy, save_ts))
        assert err < PASS_TOL, f"nonaffine/feax-pipeline relL2={err:.3e}"

    def test_nonaffine_operator_is_differentiable_in_parameter(self):
        """Autodiff gradient w.r.t. the non-affine parameter matches a
        4th-order finite difference (loose tol: float32 + exp nonlinearity)."""
        dom = init_periodic_fem(make_periodic_domain())
        block, _ = build_periodic_heat_block(dom, "nonaffine")

        def scalar(logk_val):
            A = block.operator_fn(0.0, {"logk": jnp.asarray(logk_val).reshape(())})
            return jnp.sum(jnp.asarray(A) ** 2)

        x0 = float(np.log(NU))
        g = float(jax.grad(scalar)(x0))

        # 4th-order central stencil; step scaled to |x0|, sized for float32.
        h = 1e-2 * max(abs(x0), 1.0)
        fd = float((-scalar(x0 + 2 * h) + 8 * scalar(x0 + h) - 8 * scalar(x0 - h) + scalar(x0 - 2 * h)) / (12 * h))
        rel = abs(g - fd) / (abs(fd) + 1e-8)
        # float32 + exp(logk) curvature: a few % FD agreement confirms the
        # gradient flows through the non-affine InternalVars path correctly.
        assert rel < 5e-2, f"autodiff {g:.6e} vs FD {fd:.6e} (rel {rel:.3e})"


# ============================================================
# Affine and non-affine agree (same physics)
# ============================================================


class TestAffineNonAffineConsistency:
    def test_affine_and_nonaffine_give_same_field(self):
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        dom_a = init_periodic_fem(make_periodic_domain())
        block_a, args_a = build_periodic_heat_block(dom_a, "affine")
        ys_a = solve_diffrax(dom_a, block_a, args_a, save_ts)

        dom_n = init_periodic_fem(make_periodic_domain())
        block_n, args_n = build_periodic_heat_block(dom_n, "nonaffine")
        ys_n = solve_diffrax(dom_n, block_n, args_n, save_ts)

        # exp(log 0.1) == 0.1 == nu, so the two operators are identical physics.
        assert relative_l2(ys_a, ys_n) < 1e-3


# ============================================================
# Mesh-compatibility guard
# ============================================================


class TestPeriodicMeshGuard:
    def test_unstructured_mesh_rejected_for_periodic(self):
        """Periodic identification needs node-matched opposite faces; an
        unstructured mesh must be rejected rather than silently mismatched."""
        dom = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),  # unstructured pygmsh
            time=(0.0, T_END, N_TIME),
            compute_mesh_connectivity=False,
        )
        with pytest.raises((ValueError, KeyError)):
            dom.init_fem(
                element_type="TRI3",
                quad_degree=3,
                bcs=[dom.periodic(("left", "right"), ("bottom", "top"))],
                fem_solver=True,
            )
