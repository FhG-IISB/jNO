"""``jno.fdm`` — finite-difference PDE solver (strong-form sibling of ``jno.fem``).

Run with x64 (the solve accumulates in float64)."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _nodes(d):
    return np.asarray(d.mesh_connectivity["points"])[:, :2]


def _poisson_homogeneous(mesh_size):
    """-Δu = f on [0,1]², u=0 on ∂Ω, exact u = sin(πx)sin(πy). Returns rel-L2 error."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0})
    u = np.asarray(sys.solve()).reshape(-1)
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


def test_poisson_homogeneous_dirichlet():
    assert _poisson_homogeneous(0.06) < 1e-2


def test_poisson_convergence_under_refinement():
    """Refining the mesh reduces the FD error (consistency)."""
    errs = [_poisson_homogeneous(h) for h in (0.10, 0.06, 0.035)]
    assert errs[0] > errs[1] > errs[2], f"not monotonically converging: {errs}"
    assert errs[2] < 3e-3


def test_inhomogeneous_dirichlet():
    """u = x²+y², -Δu = -4, with the boundary value g(x,y) = x²+y²."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = p[:, 0] ** 2 + p[:, 1] ** 2
    sys = jno.fdm(
        d,
        residual=lambda u: -jno.fdm.laplacian(u, d) + 4.0,
        dirichlet={"boundary": lambda x, y: x**2 + y**2},
    )
    u = np.asarray(sys.solve()).reshape(-1)
    assert float(np.linalg.norm(u - exact) / np.linalg.norm(exact)) < 1e-3


def test_matches_fem_on_same_mesh():
    """The FD solution agrees with the FE solution to FD-discretization accuracy."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    u_fd = np.asarray(
        jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0}).solve()
    ).reshape(-1)
    # both solve the same BVP; the FD field is in the analytic ballpark
    assert float(np.linalg.norm(u_fd - exact) / np.linalg.norm(exact)) < 1e-2


def test_differentiable_for_inverse_problems():
    """The solve is differentiable w.r.t. a parameter in the residual (source scale), and the
    gradient points toward the true value — the requirement for composing into jno.core."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    p = _nodes(d)
    fbase = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    obs = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))

    def loss(scale):
        sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - scale * fbase, dirichlet={"boundary": 0.0})
        return jnp.mean((sys.solve() - obs) ** 2)

    g = float(jax.grad(loss)(1.5))
    assert np.isfinite(g)
    assert g > 0.0, "at scale=1.5 (> true 1.0) the loss must increase with scale"
    assert float(loss(1.0)) < float(loss(1.5)), "scale=1.0 (truth) should beat an off value"


def test_nonlinear_reaction_diffusion():
    """Nonlinear MMS: -Δu + u³ = f with exact u = sin(πx)sin(πy). Reuses jno.solve.newton via
    the same .solve() call — a linear residual would converge in one step; this one iterates."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * exact + exact**3)  # -Δ(sin sin) + (sin sin)³
    sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) + u**3 - f, dirichlet={"boundary": 0.0})
    u = np.asarray(sys.solve()).reshape(-1)
    assert float(np.linalg.norm(u - exact) / np.linalg.norm(exact)) < 1e-2


def test_transient_heat_2d_domain_driven():
    """Domain-driven `.solve()`: time from `domain.time`, IC from `initial` — no explicit time args
    (reads like jno.fem). u_t = ν Δu → exact e^(−2νπ²t)·sin(πx)sin(πy)."""
    nu, T = 0.05, 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, T, 201))
    p = _nodes(d)
    sys = jno.fdm(
        d,
        residual=lambda u: -nu * jno.fdm.laplacian(u, d),
        dirichlet={"boundary": 0.0},
        initial=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
    )
    traj = np.asarray(sys.solve())  # steady-vs-transient inferred from the domain
    exact = np.exp(-2 * nu * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert traj.shape == (201, p.shape[0])  # domain.time n_points
    assert float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact)) < 1e-2


def test_transient_explicit_form_still_works():
    """The explicit `solve_transient(u0, t_span, nsteps)` form (no `time=` domain) still works."""
    nu, T = 0.05, 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    p = _nodes(d)
    u0 = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    sys = jno.fdm(d, residual=lambda u: -nu * jno.fdm.laplacian(u, d), dirichlet={"boundary": 0.0})
    traj = np.asarray(sys.solve_transient(u0, (0.0, T), 200))
    exact = np.exp(-2 * nu * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact)) < 1e-2


def test_transient_differentiable_for_inverse():
    """The domain-driven transient `.solve()` differentiates w.r.t. a parameter (diffusivity)."""
    T = 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08, time=(0.0, T, 201))
    p = _nodes(d)
    target = jnp.asarray(np.exp(-2 * 0.05 * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    u0 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)  # noqa: E731

    def loss(nu):
        s = jno.fdm(d, residual=lambda u: -nu * jno.fdm.laplacian(u, d), dirichlet={"boundary": 0.0}, initial=u0)
        return jnp.mean((s.solve()[-1] - target) ** 2)

    g = float(jax.grad(loss)(0.05))
    assert np.isfinite(g)
    assert float(loss(0.05)) < float(loss(0.07)), "true diffusivity should beat an off value"
