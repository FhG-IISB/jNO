"""Periodic ties composed with a runtime-parametric steady-linear FEM (Phase 3 of the compose work).

A runtime parameter makes the steady system ``A(θ)x = b(θ)``; ``jno.fem`` returns a ``FemLinearSystem``
so ``crux`` can recover ``θ`` from data (``∂u/∂θ`` via implicit diff through the user's ``solve_fn``).
A periodic tie reduces the system. The two compose only if the reduction runs **per call**, *after*
``A(θ)`` is re-formed: ``u = P · solve(PᵀA(θ)P, Pᵀb(θ))`` -- a static reduction would be silently
re-overwritten when ``operator_fn`` re-evaluates the operator. These tests pin: the reduced parametric
operator, agreement with the non-parametric periodic solve, and full recovery of ``θ`` through it.

Run with x64 (FEM assembly/solves are float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
KAPPA_TRUE = 0.7  # the diffusion coefficient the data is generated at
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})  # global FEM solve -> crux needs a driver domain


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _kappa(start, lr=5e-2):
    k = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="kappa")
    k.initialize(jax.nn.initializers.constant(start))
    k.dtype(jnp.float64)
    k.optimizer(optax.adam(lr))
    return k


def _periodic_fem(kappa, mesh_size=0.07):
    """Periodic-in-x, Dirichlet-in-y reaction-diffusion ``κ(-Δu) + u = f`` with a *fixed* source built
    for ``κ = KAPPA_TRUE`` and manufactured ``u* = cos(2πx) sin(πy)`` (``-Δu* = 5π² u*``). ``kappa`` may
    be a plain float (non-parametric) or a ``jno.np.parameter`` (the runtime-parametric system)."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = (5 * PI**2 * KAPPA_TRUE + 1.0) * jno.np.cos(2 * PI * xi) * jno.np.sin(PI * yi)
    weak = kappa * (ui.x * vi.x + ui.y * vi.y) + ui * vi - f * vi
    return jno.fem([weak, u(xb, yb) - 0.0, u(xt, yt) - 0.0, u(xl, yl) - u(xr, yr)])


def test_periodic_parametric_operator_is_reduced():
    """The parametric periodic problem yields a parametric ``FemLinearSystem`` and a periodic reduction
    that eliminates the slave-face DOFs (previously this combination raised NotImplementedError)."""
    fem = _periodic_fem(_kappa(start=1.0))
    assert fem.operator.is_parametric, "a runtime κ must produce a parametric FemLinearSystem"
    assert list(fem.operator.runtime_parameter_exprs) == ["kappa"]
    assert fem._periodic is not None, "the u(left)-u(right) tie must reduce the parametric system"
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the tie must eliminate the slave-face DOFs"


def test_periodic_parametric_forward_matches_nonparametric():
    """The per-call reduction inside FemLinearSystem.solve must give the *same* field as the eager
    reduction of the non-parametric periodic solve, evaluated at the same κ -- and recover u*."""
    u_ref = np.asarray(_periodic_fem(KAPPA_TRUE).solve())  # non-parametric periodic solve (eager reduce)
    u_node = _periodic_fem(_kappa(start=KAPPA_TRUE)).solve()  # parametric, initialized AT truth
    crux = jno.core([(u_node - u_ref).mse], domain=_DUMMY)
    u_par = np.asarray(crux.eval([u_node])).reshape(-1)  # forward eval at the initial κ (no optimization)
    assert np.allclose(u_par, u_ref.reshape(-1), atol=1e-8), "per-call reduction disagrees with eager reduction"

    pts = np.asarray(_periodic_fem(KAPPA_TRUE).points)
    u_star = np.cos(2 * PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    rel = float(np.linalg.norm(u_ref.reshape(-1) - u_star) / np.linalg.norm(u_star))
    assert rel < 2e-2, f"periodic parametric forward does not recover u*: rel-L2 {rel:.3e}"


def test_periodic_parametric_recovers_kappa():
    """Gradient check: recover κ from full-field data through the reduced parametric solve. The
    gradient ∂u/∂κ must flow through PᵀA(κ)P; starting far from truth, adam must reach KAPPA_TRUE."""
    u_obs = np.asarray(_periodic_fem(KAPPA_TRUE).solve())  # FEM-consistent clean data
    kappa = _kappa(start=1.5)  # start far from the truth 0.7
    u_node = _periodic_fem(kappa).solve()
    crux = jno.core([(u_node - u_obs).mse], domain=_DUMMY)
    crux.solve(200)
    rec = float(np.asarray(crux.eval([kappa])).reshape(-1)[0])
    assert abs(rec - KAPPA_TRUE) < 0.05, f"κ not recovered through the reduced parametric solve: {rec:.4f}"
