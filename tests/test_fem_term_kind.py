"""Tests for structural term classification (jno.utils.solver.term_kind).

Covers the discriminating cases that distinguish local (pointwise reaction/mass) terms from
global (neighbour-coupling diffusion/advection) terms — including the temporal-vs-spatial trap
(``u.t`` is a derivative but spatially local) and the multifield bilinear reaction (the PEB target).
"""

import jax
import pytest

import jno
from jno.utils.solver.term_kind import classify_term


@pytest.fixture
def ctx():
    """Small 2D+time domain with two fields; x64 set per-test (FEM assembly is float64)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    dom = (
        jno.Shape.rect(0.0, 0.0, 1.0, 1.0)
        .structured(n=8)
        .domain(
            time=(0.0, 1.0, 3),
            compute_mesh_connectivity=False,
        )
    )
    u, v = dom.fem_symbols()
    A, qA = dom.fem_symbols(names=("A", "qA"))
    xi, yi, ti = dom.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    Ai = A.bind(x=xi, y=yi, t=ti)
    try:
        yield dom, ui, vi, Ai
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_mass_term_is_local(ctx):
    """u.t * v : a temporal derivative, but spatially pointwise (the trap)."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, ui.t * vi)
    assert k.is_local
    assert k.time_order == 1
    assert k.trial_channel == "value"
    assert k.test_channel == "value"


def test_diffusion_is_global(ctx):
    """nu * grad(u).grad(v) : spatial gradient on both sides → global."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, 0.1 * (ui.x * vi.x + ui.y * vi.y))
    assert not k.is_local
    assert k.trial_channel == "grad"
    assert k.test_channel == "grad"


def test_advection_is_global(ctx):
    """b * u.x * v : grad on trial, value on test → still global (couples neighbours)."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, ui.x * vi)
    assert not k.is_local
    assert k.trial_channel == "grad"
    assert k.test_channel == "value"


def test_bilinear_reaction_is_local_and_nonlinear(ctx):
    """k * A * u * v : two trial fields, no gradient → local + nonlinear (the PEB reaction)."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, 5.0 * Ai * ui * vi)
    assert k.is_local
    assert not k.linear
    assert k.trial_channel == "value"
    assert k.test_channel == "value"


def test_linear_reaction_is_local_and_linear(ctx):
    """c * u * v : single trial value, no gradient → local + linear (mass-like reaction)."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, 0.7 * ui * vi)
    assert k.is_local
    assert k.linear


def test_source_has_no_trial(ctx):
    """f * v : no trial field at all → trial channel 'none', still local."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, 2.0 * vi)
    assert k.trial_channel == "none"
    assert k.test_channel == "value"
    assert k.is_local


def test_full_reaction_diffusion_term_is_global(ctx):
    """A whole transient diffusion-reaction residual (mass + diffusion + reaction) contains a
    spatial gradient, so the *combined* term is global — terms must be split additively before
    classification for the local part to be isolated (what the routing pass will do)."""
    dom, ui, vi, Ai = ctx
    k = classify_term(dom, ui.t * vi + 0.1 * (ui.x * vi.x) + 5.0 * Ai * ui * vi)
    assert not k.is_local  # the diffusion sub-term makes the lumped term global
    assert k.time_order == 1


def test_test_channel_matches_vpinn_leaf_predicates(ctx):
    """The classifier's test channel must agree with the VPINN leaf predicates so the two
    notions of value-vs-grad test can't drift."""
    dom, ui, vi, Ai = ctx
    from jno.utils.solver.weak_form_helpers import is_test_grad, is_test_value

    assert is_test_value(vi.expr)  # a plain test value
    assert classify_term(dom, ui * vi).test_channel == "value"
    assert is_test_grad(vi.x.expr)  # a test gradient
    assert classify_term(dom, ui.x * vi.x).test_channel == "grad"


def test_term_kinds_accessor_breaks_down_a_transient_pde():
    """fem.term_kinds splits a transient diffusion-reaction equation into its sub-terms and
    classifies each: mass (local), diffusion (global), reaction (local)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        dom = (
            jno.Shape.rect(0.0, 0.0, 1.0, 1.0)
            .structured(n=8)
            .domain(
                time=(0.0, 0.1, 3),
                compute_mesh_connectivity=False,
            )
        )
        dom.tag("bnd", lambda x, y: (x < 1e-6) | (x > 1 - 1e-6) | (y < 1e-6) | (y > 1 - 1e-6))
        u, v = dom.fem_symbols()
        xi, yi, ti = dom.variable("interior", split=True)
        ci = dom.variable("initial", split=True)
        xb, yb, _ = dom.variable("bnd", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        eq = ui.t * vi + 0.1 * ui.x * vi.x + 0.1 * ui.y * vi.y + 2.0 * ui * vi
        fem = jno.fem([eq, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.0])

        kinds = fem.term_kinds
        assert kinds is not None
        assert all(k.support == "volume" for k in kinds)
        assert any(k.is_local and k.time_order == 1 for k in kinds)  # mass term
        assert any(not k.is_local for k in kinds)  # diffusion
        assert any(k.is_local and k.time_order == 0 and k.linear for k in kinds)  # linear reaction
    finally:
        jax.config.update("jax_enable_x64", prev)
