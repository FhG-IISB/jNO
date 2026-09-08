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

import jno

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


@pytest.fixture(autouse=True)
def _x64():
    """x64 per TEST, saved and restored -- never at module scope.

    Setting it at import runs for every module in the selection and cannot be undone, which
    tests/test_x64_isolation.py exists to forbid.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
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


def _nu_t_3d(model, G):
    """Same reading, in 3-D: u_i = G_ij x_j on a unit cube, so grad(u) = G exactly on every tet."""
    d = jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.7).domain()
    co = d.variable("interior", split=True)
    ax = [co[0], co[1], co[2]]
    u, v = d.fem_symbols(value_shape=(3,), names=("u3", "v3"), order=1)
    s, w = d.fem_symbols(names=("s3", "w3"), order=1)
    si, wi = s.bind(x=ax[0], y=ax[1], z=ax[2]), w.bind(x=ax[0], y=ax[1], z=ax[2])
    fem = jno.fem([inner_(grad(u, ax), grad(v, ax), n_contract=2), si * wi - model(grad(u, ax)) * wi])
    pts = np.asarray(fem.points)[:, :3]
    uk = np.zeros(fem.dofs)
    bu = fem.blocks[fem.block_index(u)]
    uk[bu.start : bu.stop] = (pts @ np.asarray(G).T).reshape(-1)
    r = np.asarray(fem.residual(uk))
    bs = fem.blocks[fem.block_index(s)]
    return -float(r[bs.start : bs.stop].sum())


@pytest.mark.parametrize("name", ["smagorinsky", "vreman", "wale"])
def test_each_model_is_dimension_generic(name):
    """The models are written through second invariants precisely so one spelling serves 2-D and 3-D.
    In 2-D that is cheap to satisfy by accident -- several distinct expressions collapse onto the same
    number on a 2x2 tensor -- so the claim is only worth anything when checked on a full 3x3 gradient,
    including a case with no symmetry at all."""
    rng = np.random.default_rng(3)
    cases = {
        "shear": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "random": rng.normal(scale=2.0, size=(3, 3)),
        "incompressible strain": np.diag([2.0, -1.0, -1.0]),
    }
    models3 = {"smagorinsky": smagorinsky, "vreman": vreman, "wale": lambda g: wale(g, 3)}
    for label, G in cases.items():
        got = _nu_t_3d(models3[name], G)
        # abs= as well as rel=: where the exact answer is 0 (Vreman in shear) the invariant is a
        # difference of equal numbers, so it lands on the cancellation floor -- 7e-12 here -- and a
        # bare rel= against 0 would demand an exactness float64 cannot deliver. nu_t is O(1e-3), so
        # 1e-10 is still five orders below anything the model reports.
        assert got == pytest.approx(_models_numpy(G, dim=3)[name], rel=1e-9, abs=1e-10), f"{name} on {label}: {got}"


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


# --------------------------------------------------------------------------------------------
# The models INSIDE a solve. The tests above evaluate them on prescribed gradients; these run them
# in a Newton loop, where two further things can go wrong that arithmetic alone cannot show.
# --------------------------------------------------------------------------------------------
def _solved_shear(model):
    """Solve a stabilised P1/P1 Couette flow with the model active; return (max nu_t, max |u|).

    u = (y, 0) on the boundary is linear and an exact solution, so the interior reproduces it and the
    interior gradient is SIMPLE SHEAR -- the structure Vreman and WALE are built to vanish on. A
    scalar block projects nu_t so it can be read; it is present for every model including `none`, so
    the systems compared have identical structure.
    """
    NU = 1e-2
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain()
    d.tag("all", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0))
    xi, yi = d.variable("interior", split=True)[:2]
    xa, ya = d.variable("all", split=True)[:2]
    xn, yn = d.variable("ppin", split=True)[:2]
    ax = [xi, yi]
    u, v = d.fem_symbols(value_shape=(2,), names=("u_s", "v_s"), order=1)
    p, q = d.fem_symbols(names=("p_s", "q_s"), order=1)
    nut, w = d.fem_symbols(names=("nut_s", "w_s"), order=1)
    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    nb, wb = nut.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    gu, gv = grad(u, ax), grad(v, ax)

    # LAGGED. nu_t is a square root, so its slope at u = 0 is infinite: unlagged, Newton diverges from
    # a rest state outright. That this solve converges at all is half of what these tests check.
    nu_t = jno.lag(0.0 if model is None else model(gu))
    adv = lambda gw, ww: inner_(gw, ww, n_contract=1)  # noqa: E731
    mom = inner_(adv(gu, ub), vv, n_contract=1) + (NU + nu_t) * inner_(gu, gv, n_contract=2) - pp * trace(gv)
    cont = -qq * trace(gu)
    G = d.cell_metric
    gG = lambda a: inner_(a, inner_(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag((gG(ub) + 36.0 * NU**2 * inner_(G, G, n_contract=2)) ** -0.5)
    r_m = adv(gu, ub) - NU * jno.np.laplacian(u, ax) + grad(p, ax)
    mom = mom + tau * inner_(adv(gv, ub), r_m, n_contract=1)
    cont = cont - tau * inner_(grad(q, ax), r_m, n_contract=1)

    fem = jno.fem(
        [
            mom,
            cont,
            nb * wb - nu_t * wb,
            u(xa, ya)[0] - ya,
            u(xa, ya)[1] - 0.0,
            p(xn, yn) - 0.0,
        ]
    )
    sol = np.asarray(
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-10, atol=1e-10), linear=jno.solve.lu(backend="host"))
    )
    bu, bn = fem.blocks[fem.block_index(u)], fem.blocks[fem.block_index(nut)]
    return float(np.abs(sol[bn.start : bn.stop]).max()), float(np.abs(sol[bu.start : bu.stop]).max())


def test_on_a_solved_shear_flow_the_good_models_vanish_and_smagorinsky_does_not():
    """The discriminating property, now inside a Newton solve rather than on a prescribed gradient.

    Judged RELATIVE to Smagorinsky: Vreman's invariant is a difference of nearly equal numbers, so
    where the answer is exactly zero it lands on the cancellation floor rather than on 0.
    """
    smag, _ = _solved_shear(smagorinsky)
    assert smag > 1e-6, f"smagorinsky should be clearly non-zero in shear, got {smag:.3e}"
    for name, model in (("vreman", vreman), ("wale", lambda g: wale(g, 2))):
        nut, _u = _solved_shear(model)
        assert nut < 1e-3 * smag, f"{name} must vanish in a solved shear flow: {nut:.3e} vs smagorinsky {smag:.3e}"


def test_a_lagged_eddy_viscosity_converges_and_leaves_the_shear_profile_exact():
    """Two regressions in one. The solve must CONVERGE (nu_t is a sqrt; unlagged its tangent is
    singular at rest), and since nu_t is spatially constant in Couette it must not bend the profile --
    a constant viscosity leaves a linear shear profile alone, whatever its value."""
    base_nut, base_u = _solved_shear(None)
    assert base_nut == pytest.approx(0.0, abs=1e-14)
    for model in (vreman, lambda g: wale(g, 2), smagorinsky):
        nut, umax = _solved_shear(model)
        assert np.isfinite(nut) and nut >= 0.0, f"nu_t must be finite and non-negative, got {nut}"
        assert umax == pytest.approx(base_u, rel=1e-6), f"the shear profile moved: {umax} vs {base_u}"
