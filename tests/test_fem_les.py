"""LES eddy viscosities are FORMULAS -- no library API, per the FEM contract.

Three algebraic (zero-equation) subgrid models, each a function of the resolved velocity gradient
``g[i,j] = du_i/dx_j`` and a filter width ``delta``:

* **Smagorinsky** (1963)  nu_t = (Cs d)^2 |S|,  |S| = sqrt(2 S:S), S = sym(g);
* **Vreman**, Phys. Fluids 16 (2004) 3670, eq. (5)  nu_t = c sqrt(B_beta / (g:g));
* **WALE**, Nicoud & Ducros, Flow Turb. Combust. 62 (1999) 183, eq. (13).

The test that matters is the one that separates them. In **pure laminar shear** a subgrid model must
produce ZERO eddy viscosity -- there is nothing to model. Smagorinsky does not: |S| is non-zero in any
shear, so it invents turbulence throughout a laminar boundary layer, which is why it needs Van Driest
damping. Vreman and WALE vanish identically there, by construction. That single case is worth more
than a channel-flow benchmark for pinning that the algebra is right.

Everything is written with full contractions (`inner`, `trace`, `einsum` to a scalar) and `sym`, so
the models are DIMENSION-GENERIC: both `B_beta` and `Sd:Sd` are second invariants, expressed through
tr(.) and tr(.^2) rather than by naming components.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402

inner_, grad, trace, sym, einsum, sqrt, where = (
    jno.np.inner,
    jno.np.grad,
    jno.np.trace,
    jno.np.sym,
    jno.np.einsum,
    jno.np.sqrt,
    jno.np.where,
)

# Both invariants below are non-negative in exact arithmetic and are computed as a DIFFERENCE of two
# nearly equal numbers, so round-off can push them just below zero -- and a negative one is not a
# small error but a NaN, via sqrt() and **1.5. Measured in pure shear: B_beta lands at 1.4e-20 where
# it should be 0, from a relative cancellation of 2.7e-16. Clamping is part of the model, not a tidy-up.
_clamp = lambda z: where(z > 0.0, z, 0.0)  # noqa: E731

DELTA, CS, CV, CW = 0.1, 0.17, 0.07, 0.325
EPS = 1e-30


# --------------------------------------------------------------------------------------------
# The models, written the way a user would write them in a weak form.
# --------------------------------------------------------------------------------------------
def smagorinsky(g, d=DELTA, cs=CS):
    return (cs * d) ** 2 * sqrt(2.0 * inner_(sym(g), sym(g), n_contract=2))


def vreman(g, d=DELTA, c=CV):
    """nu_t = c sqrt(B_beta / (alpha:alpha)) with beta = d^2 g g^T.

    B_beta is the second invariant of beta, (tr(beta)^2 - tr(beta^2))/2 -- so it needs no component
    names and works in 2-D and 3-D alike. tr(beta) = d^2 g:g, and tr(beta^2) = d^4 tr(g g^T g g^T)."""
    tr_b = d**2 * inner_(g, g, n_contract=2)
    tr_b2 = d**4 * einsum("...ik,...jk,...jl,...il->...", g, g, g, g)
    b_beta = _clamp(0.5 * (tr_b**2 - tr_b2))
    return c * sqrt(b_beta / (inner_(g, g, n_contract=2) + EPS) + EPS)


def wale(g, dim, d=DELTA, cw=CW):
    """nu_t = (Cw d)^2 (Sd:Sd)^{3/2} / ((S:S)^{5/2} + (Sd:Sd)^{5/4}).

    Sd = sym(g^2) - I tr(g^2)/dim, so Sd:Sd = sym(g^2):sym(g^2) - tr(g^2)^2/dim -- the deviator is
    never formed, which also sidesteps `identity(n) * <per-qp scalar>` broadcasting."""
    tr_g2 = einsum("...ij,...ji->...", g, g)  # tr(g @ g)
    g2_g2T = einsum("...ik,...kj,...il,...lj->...", g, g, g, g)  # tr(g^2 (g^2)^T)
    g2_g2 = einsum("...ik,...kj,...jl,...li->...", g, g, g, g)  # tr(g^2 g^2)
    sd = _clamp(0.5 * (g2_g2T + g2_g2) - tr_g2**2 / dim)
    ss = inner_(sym(g), sym(g), n_contract=2)
    return (cw * d) ** 2 * (sd**1.5 / (ss**2.5 + sd**1.25 + EPS))


def _models_numpy(g, dim=2):
    """Textbook oracle, written with explicit transposes and an explicit deviator."""
    S = 0.5 * (g + g.T)
    smag = (CS * DELTA) ** 2 * np.sqrt(2.0 * np.sum(S * S))
    b = DELTA**2 * (g @ g.T)
    b_beta = 0.5 * (np.trace(b) ** 2 - np.trace(b @ b))
    gg = np.sum(g * g)
    vre = CV * np.sqrt(max(b_beta, 0.0) / gg) if gg > 1e-300 else 0.0
    g2 = g @ g
    Sd = 0.5 * (g2 + g2.T) - np.eye(dim) * np.trace(g2) / dim
    sd, ss = np.sum(Sd * Sd), np.sum(S * S)
    wl = (CW * DELTA) ** 2 * (sd**1.5 / (ss**2.5 + sd**1.25)) if (ss + sd) > 1e-300 else 0.0
    return {"smagorinsky": smag, "vreman": vre, "wale": wl}


# --------------------------------------------------------------------------------------------
def _nu_t(model, a, b, c, e):
    """Evaluate a model on the exact linear field u = (a x + b y, c x + e y).

    Linear, so P1 represents it exactly and grad(u) = [[a, b], [c, e]] on every element. The value is
    read off an assembled residual at a prescribed dof vector -- no solve, so the number depends on
    the model algebra alone. The scalar block's equation is ``s*w - nu_t*w``, which at s = 0 has
    residual ``-nu_t * integral(w)``; P1 test functions are a partition of unity, so summing that
    block over the unit square returns ``-nu_t`` for a spatially constant nu_t.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    xi, yi = d.variable("interior", split=True)[:2]
    ax = [xi, yi]
    u, v = d.fem_symbols(value_shape=(2,), names=("u_l", "v_l"), order=1)
    s, w = d.fem_symbols(names=("s_l", "w_l"), order=1)
    si, wi = s.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    expr = model(grad(u, ax))
    fem = jno.fem([inner_(grad(u, ax), grad(v, ax), n_contract=2), si * wi - expr * wi])
    pts = np.asarray(fem.points)
    uk = np.zeros(fem.dofs)
    bu = fem.blocks[fem.block_index(u)]
    uv = np.stack([a * pts[:, 0] + b * pts[:, 1], c * pts[:, 0] + e * pts[:, 1]], axis=-1)
    uk[bu.start : bu.stop] = uv.reshape(-1)
    r = np.asarray(fem.residual(uk))
    bs = fem.blocks[fem.block_index(s)]
    return -float(r[bs.start : bs.stop].sum())


MODELS = {"smagorinsky": smagorinsky, "vreman": vreman, "wale": lambda g: wale(g, 2)}
SHEAR = (0.0, 1.0, 0.0, 0.0)  # u = (y, 0): g = [[0,1],[0,0]], the laminar-shear case
GENERAL = (1.0, 2.0, 3.0, -1.0)  # u = (x+2y, 3x-y): g = [[1,2],[3,-1]], nothing special


@pytest.mark.parametrize("name", ["vreman", "wale"])
def test_a_good_subgrid_model_vanishes_in_pure_laminar_shear(name):
    """THE discriminating property. There is no subgrid turbulence in a laminar shear layer, so a
    model that reports eddy viscosity there is adding dissipation that is not physics."""
    got = _nu_t(MODELS[name], *SHEAR)
    # Relative to Smagorinsky on the SAME gradient, which is the meaningful scale: "vanishes" means
    # orders below the model that does not, not below an absolute floor the cancellation cannot reach.
    ref = _nu_t(smagorinsky, *SHEAR)
    assert abs(got) < 1e-6 * ref, f"{name} must vanish in pure shear: {got:.3e} vs smagorinsky {ref:.3e}"


def test_smagorinsky_does_not_vanish_and_that_is_the_known_defect():
    """The control. Smagorinsky reports a non-zero eddy viscosity in the same laminar shear -- this
    is why it needs Van Driest wall damping, and why the test above is worth having."""
    got = _nu_t(smagorinsky, *SHEAR)
    assert got > 1e-6, f"smagorinsky should be non-zero in pure shear, got {got:.3e}"
    assert got == pytest.approx(_models_numpy(np.array([[0.0, 1.0], [0.0, 0.0]]))["smagorinsky"], rel=1e-10)


@pytest.mark.parametrize("name", ["smagorinsky", "vreman", "wale"])
def test_each_model_matches_a_textbook_numpy_oracle(name):
    """On a general gradient, against the same model written with explicit transposes and deviator --
    so the invariant-based spellings above are checked, not just self-consistent."""
    g = np.array([[GENERAL[0], GENERAL[1]], [GENERAL[2], GENERAL[3]]])
    got = _nu_t(MODELS[name], *GENERAL)
    assert got == pytest.approx(_models_numpy(g)[name], rel=1e-9), f"{name}: {got} vs oracle"


@pytest.mark.parametrize("name", ["smagorinsky", "vreman", "wale"])
def test_the_eddy_viscosity_is_never_negative(name):
    """A negative nu_t is anti-diffusion: it would add energy and destabilise the march. Checked on a
    spread of gradients, including the degenerate ones (zero, pure shear, pure rotation)."""
    rng = np.random.default_rng(0)
    cases = [(0.0, 0.0, 0.0, 0.0), SHEAR, (0.0, -1.0, 1.0, 0.0), GENERAL]
    cases += [tuple(rng.normal(scale=3.0, size=4)) for _ in range(4)]
    for cse in cases:
        got = _nu_t(MODELS[name], *cse)
        assert got >= -1e-12, f"{name} went negative ({got:.3e}) at g={cse}"
        assert np.isfinite(got), f"{name} non-finite at g={cse}"
