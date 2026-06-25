"""Lightweight tests for periodic BCs and affine / non-affine runtime parameters.

These lock in the behaviour verified by hand in check_periodic_bc.py:

  * periodic reduction builds a prolongation P and a reduced (n_red < n_full)
    semidiscrete system, on a structured (periodic-compatible) mesh;
  * an AFFINE runtime parameter (nu * grad u . grad phi) is exposed through the
    operator and reproduces the analytical periodic-heat decay when stepped with
    the default backward-Euler integrator;
  * a NON-AFFINE runtime parameter (exp(logk) * grad u . grad phi, parameter
    inside a nonlinear function) routes through the native non-affine operator
    path, is reported in the block metadata, and -- because exp(log 0.1) = 0.1 --
    reproduces the SAME analytical solution as the affine case.

The analytical reference is the separable periodic heat mode

    u(x,y,t) = exp(-8 pi^2 nu t) sin(2 pi x) sin(2 pi y),   nu = 0.1.
"""

import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax
import jax.numpy as jnp
import numpy as np

import jno
import jno.jnp_ops as jnn
from jno.utils.solver.backend_blocks import _default_transient_integrate


@pytest.fixture(autouse=True)
def _x64_off():
    """These periodic tests run in float32 by design (loose-tol FD / backward-Euler
    checks). Set x64 False *per-test* with save/restore so the global flag never leaks
    into FEM modules co-run after this one (they need x64). See the symmetric `_x64`
    fixture in test_fem_inverse.py."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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


_FACES = {
    "left": lambda x, y: x < 1e-6,
    "right": lambda x, y: x > 1 - 1e-6,
    "bottom": lambda x, y: y < 1e-6,
    "top": lambda x, y: y > 1 - 1e-6,
}


def make_periodic_domain():
    """Structured (periodic-compatible) rectangular domain on [0,1]^2, with each face named via
    ``domain.tag`` -- required for multi-direction periodicity so the shared corners are recovered."""
    dom = jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=N_GRID, ny=N_GRID),
        time=(0.0, T_END, N_TIME),
        compute_mesh_connectivity=False,
    )
    for nm, pred in _FACES.items():
        dom.tag(nm, pred)
    return dom


def build_periodic_heat_block(dom, kind):
    """Doubly-periodic heat ``u_t = nu * lap u`` (periodic in x and y), authored through ``jno.fem``
    (the sole entry -- no ``init_fem`` / ``weak.assemble``). ``kind`` in {'affine', 'nonaffine'};
    returns ``(block, args)`` where ``block = fem.operator`` is the periodic-reduced time block."""
    u, phi = dom.fem_symbols()
    xi, yi, ti = dom.variable("interior", split=True)
    ci = dom.variable("initial", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    diffusion = ui.x * vi.x + ui.y * vi.y
    ic = jnn.sin(TWO_PI * ci[0]) * jnn.sin(TWO_PI * ci[1])

    if kind == "affine":
        nu = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="nu")
        nu.initialize(jax.nn.initializers.constant(NU))
        weak = ui.t * vi + nu * diffusion
        args = {"nu": NU}
    elif kind == "nonaffine":
        logk = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="logk")
        logk.initialize(jax.nn.initializers.constant(float(np.log(NU))))
        weak = ui.t * vi + jno.np.exp(logk) * diffusion  # parameter inside exp
        args = {"logk": float(np.log(NU))}
    else:
        raise ValueError(kind)

    fem = jno.fem([weak, u(xl, yl) - u(xr, yr), u(xb, yb) - u(xt, yt), u(ci[0], ci[1]) - ic])
    return fem.operator, args


def analytical(mesh_xy, save_ts):
    spatial = np.sin(TWO_PI * mesh_xy[:, 0]) * np.sin(TWO_PI * mesh_xy[:, 1])
    decay = np.exp(-8.0 * np.pi**2 * NU * np.asarray(save_ts))[:, None]
    return decay * spatial[None, :]


def relative_l2(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def solve_transient(dom, block, args, save_ts):
    """Integrate the reduced periodic block with jNO's default backward-Euler stepper, then
    prolong (``u_full = P u_red``) to compare against the analytical solution. ``P`` lives on the
    block itself (the native periodic reduction attaches it as ``block.prolongation``)."""
    P = jnp.asarray(block.prolongation.todense())  # prolongation is a BCOO selection matrix
    ys = _default_transient_integrate(block, args, jnp.asarray(save_ts))
    return np.asarray(jnp.asarray(ys) @ P.T)


# ============================================================
# Periodic reduction structure
# ============================================================


class TestPeriodicReductionStructure:
    def test_block_carries_prolongation_and_reduces_dofs(self):
        dom = make_periodic_domain()
        block, _ = build_periodic_heat_block(dom, "affine")

        P = np.asarray(block.prolongation.todense())  # prolongation is a BCOO selection matrix
        n_full = int(np.asarray(dom.mesh.points).shape[0])

        assert P.ndim == 2
        assert P.shape[0] == n_full  # full rows
        assert P.shape[1] < n_full  # reduced columns
        assert block.metadata["full_state_size"] == n_full
        assert block.metadata["reduced_state_size"] == P.shape[1]
        # Each row of P selects exactly one master DOF (partition-of-unity rows).
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_periodic_block_carries_reduced_system_and_prolongation(self):
        dom = make_periodic_domain()
        block, _ = build_periodic_heat_block(dom, "affine")

        n_full = int(np.asarray(dom.mesh.points).shape[0])
        assert bool(block.metadata.get("periodic")) is True
        assert block.metadata["full_state_size"] == n_full
        assert block.metadata["reduced_state_size"] < n_full

        M = block.M  # reduced operator is BCOO (sparse periodic reduction) -- .shape works on both
        assert M.shape[0] == M.shape[1]
        assert M.shape[0] == block.metadata["reduced_state_size"]
        assert getattr(block, "prolongation", None) is not None


# ============================================================
# Affine runtime parameter
# ============================================================


class TestPeriodicAffineParameter:
    def test_affine_parameter_reported(self):
        dom = make_periodic_domain()
        block, _ = build_periodic_heat_block(dom, "affine")

        assert block.operator_fn is not None
        assert "nu" in list(block.metadata.get("runtime_parameter_names", []))

    def test_affine_periodic_matches_analytical(self):
        dom = make_periodic_domain()
        block, args = build_periodic_heat_block(dom, "affine")
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        mesh_xy = np.asarray(dom.mesh.points)[:, :2]
        err = relative_l2(solve_transient(dom, block, args, save_ts), analytical(mesh_xy, save_ts))
        assert err < PASS_TOL, f"affine relL2={err:.3e}"


# ============================================================
# Non-affine runtime parameter
# ============================================================


class TestPeriodicNonAffineParameter:
    def test_nonaffine_parameter_routed_and_reported(self):
        dom = make_periodic_domain()
        block, _ = build_periodic_heat_block(dom, "nonaffine")

        assert block.operator_fn is not None
        assert bool(block.metadata.get("nonaffine_operator")) is True
        assert "logk" in list(block.metadata.get("runtime_parameter_names", []))

    def test_nonaffine_operator_is_nonzero_at_args(self):
        dom = make_periodic_domain()
        block, args = build_periodic_heat_block(dom, "nonaffine")

        A = block.operator_fn(0.0, {k: jnp.asarray(v) for k, v in args.items()})
        n_red = block.metadata["reduced_state_size"]
        assert A.shape == (n_red, n_red)  # operator is reduced (BCOO)
        A_dense = A.todense() if hasattr(A, "todense") else jnp.asarray(A)
        assert float(jnp.linalg.norm(A_dense)) > 0.0

    def test_nonaffine_periodic_matches_analytical(self):
        dom = make_periodic_domain()
        block, args = build_periodic_heat_block(dom, "nonaffine")
        save_ts = np.linspace(0.0, T_END, N_TIME, dtype=np.float32)

        mesh_xy = np.asarray(dom.mesh.points)[:, :2]
        err = relative_l2(solve_transient(dom, block, args, save_ts), analytical(mesh_xy, save_ts))
        assert err < PASS_TOL, f"nonaffine relL2={err:.3e}"

    def test_nonaffine_operator_is_differentiable_in_parameter(self):
        """Autodiff gradient w.r.t. the non-affine parameter matches a
        4th-order finite difference (loose tol: float32 + exp nonlinearity)."""
        dom = make_periodic_domain()
        block, _ = build_periodic_heat_block(dom, "nonaffine")

        def scalar(logk_val):
            A = block.operator_fn(0.0, {"logk": jnp.asarray(logk_val).reshape(())})
            A = A.todense() if hasattr(A, "todense") else jnp.asarray(A)  # reduced op is BCOO; densify keeps the AD trace
            return jnp.sum(A**2)

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

        dom_a = make_periodic_domain()
        block_a, args_a = build_periodic_heat_block(dom_a, "affine")
        ys_a = solve_transient(dom_a, block_a, args_a, save_ts)

        dom_n = make_periodic_domain()
        block_n, args_n = build_periodic_heat_block(dom_n, "nonaffine")
        ys_n = solve_transient(dom_n, block_n, args_n, save_ts)

        # exp(log 0.1) == 0.1 == nu, so the two operators are identical physics.
        assert relative_l2(ys_a, ys_n) < 1e-3


# ============================================================
# Mesh-compatibility guard
# ============================================================


class TestPeriodicMeshGuard:
    def test_multidirection_without_tag_predicates_rejected(self):
        """Multi-direction periodicity needs each face defined via ``domain.tag(name, predicate)`` so
        the shared corners (a slave in two directions) are recovered. Auto-generated face tags (no
        predicate) cannot resolve the corners, so jno.fem rejects them loudly rather than silently
        under-identifying the corners and mis-solving."""
        dom = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),  # auto-generated left/right/... tags (no predicate)
            time=(0.0, T_END, N_TIME),
            compute_mesh_connectivity=False,
        )
        u, phi = dom.fem_symbols()
        xi, yi, ti = dom.variable("interior", split=True)
        ci = dom.variable("initial", split=True)
        xl, yl, _ = dom.variable("left", split=True)
        xr, yr, _ = dom.variable("right", split=True)
        xb, yb, _ = dom.variable("bottom", split=True)
        xt, yt, _ = dom.variable("top", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
        ic = jnn.sin(TWO_PI * ci[0]) * jnn.sin(TWO_PI * ci[1])
        with pytest.raises(NotImplementedError):
            jno.fem(
                [
                    ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),
                    u(xl, yl) - u(xr, yr),
                    u(xb, yb) - u(xt, yt),
                    u(ci[0], ci[1]) - ic,
                ]
            )
