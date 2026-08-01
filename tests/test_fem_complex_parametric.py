"""A runtime parameter recovered through a complex forward solve -- the complex *inverse* (Phase 5).

A complex weak form assembles as two real systems solved via the real-equivalent block. With a runtime
``jno.np.parameter`` the legs become parametric ``FemLinearSystem``s and ``jno.fem`` returns a
differentiable trace node: ``A(θ), b(θ)`` are re-formed and the block re-solved per call, so ``∂u/∂θ``
flows to ``crux`` (previously this raised NotImplementedError -- "the complex inverse is a follow-on").
Here the parameter is *real* (it scales the diffusion), so only the real-coefficient leg is parametric
and the imaginary leg is a constant ``(A, b)`` -- the mixed case the block solve must handle.

Run with x64 (the solution is complex128).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.trace import FemLinearSystem  # noqa: E402

PI = np.pi
KAPPA_TRUE = 0.8
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


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


def _complex_fem(kappa, mesh_size=0.08):
    """Complex Helmholtz ``κ(-Δu) + d·u = f`` (all-Neumann box), ``d = 1+0.3i``, manufactured complex
    ``u* = (1+0.5i) cos(πx) cos(πy)`` (zero normal derivative). With ``-Δu* = 2π² u*`` the source is
    ``f = (2π² KAPPA_TRUE + d) u*``. ``kappa`` is a float (forward) or a real ``jno.np.parameter``."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    d_coef = 1.0 + 0.3j
    amp = 1.0 + 0.5j
    g = jno.np.cos(PI * xi) * jno.np.cos(PI * yi)
    f = (2 * PI**2 * KAPPA_TRUE + d_coef) * amp * g
    weak = kappa * (ui.x * vi.x + ui.y * vi.y) + d_coef * (u * vi) - f * vi
    return jno.fem([weak])


def test_complex_parametric_yields_parametric_legs_and_node():
    """A real parameter in a complex form: the Re/Im legs are fused into ONE parametric
    ``FemLinearSystem`` over the real-equivalent 2n block, which carries κ and re-forms ``A(κ), b(κ)``
    per call — so ``solve()`` is a trace node, not an array, and the parameter can flow to crux.

    The unfused legs stay reachable on ``_complex_legs``, because a complex-native preconditioner (AMS)
    solves ``A_r + i·A_i`` rather than the block; the assembler registers κ on whichever legs reference
    it (here the real one — κ scales the diffusion)."""
    fem = _complex_fem(_kappa(start=1.0))
    assert fem.is_complex
    fused = fem._op
    assert isinstance(fused, FemLinearSystem) and fused.is_parametric, "the fused 2n system must carry κ"
    assert list(fused.runtime_parameter_exprs) == ["kappa"]
    n = int(np.asarray(fem.points).shape[0])
    assert fused.A.shape == (2 * n, 2 * n), f"expected the real-equivalent 2n block, got {fused.A.shape}"

    op_r, _op_i = fem._complex_legs  # retained for the complex-native (AMS) path
    assert isinstance(op_r, FemLinearSystem) and op_r.is_parametric, "the real leg must carry κ"
    assert list(op_r.runtime_parameter_exprs) == ["kappa"]
    assert not isinstance(fem.solve(), jax.Array), "a parametric complex solve must be a trace node"


def test_complex_parametric_forward_matches_nonparametric():
    """Eval the parametric node at κ_true: it must match the non-parametric complex solve and recover u*."""
    u_ref = np.asarray(_complex_fem(KAPPA_TRUE).solve())  # non-parametric complex forward
    u_node = _complex_fem(_kappa(start=KAPPA_TRUE)).solve()
    crux = jno.core([(u_node - u_ref).mae], domain=_DUMMY)
    u_par = np.asarray(crux.eval([u_node])).reshape(-1)
    assert np.allclose(u_par, u_ref.reshape(-1), atol=1e-8), "parametric complex solve disagrees with forward"

    pts = np.asarray(_complex_fem(KAPPA_TRUE).points)
    u_star = (1.0 + 0.5j) * np.cos(PI * pts[:, 0]) * np.cos(PI * pts[:, 1])
    rel = float(np.linalg.norm(u_ref.reshape(-1) - u_star) / np.linalg.norm(u_star))
    assert rel < 2e-2, f"complex forward does not recover u*: rel-L2 {rel:.3e}"
    assert float(np.abs(u_ref.imag).max()) > 0.1, "must be genuinely complex"


def test_complex_parametric_recovers_kappa():
    """Gradient check: recover the real κ from complex full-field data through the real-equivalent block.
    ∂u/∂κ must flow through the (parametric) real leg; adam from κ=1.5 must reach KAPPA_TRUE."""
    u_obs = np.asarray(_complex_fem(KAPPA_TRUE).solve())  # complex clean data
    kappa = _kappa(start=1.5)
    u_node = _complex_fem(kappa).solve()
    crux = jno.core([(u_node - u_obs).mae], domain=_DUMMY)  # mean|·| -> a real loss on a complex residual
    crux.solve(250)
    rec = float(np.asarray(crux.eval([kappa])).reshape(-1)[0])
    assert abs(rec - KAPPA_TRUE) < 0.05, f"κ not recovered through the complex inverse: {rec:.4f}"
