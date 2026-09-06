"""Restricting an internal state to a REGION — ``state.evolves(formula, region=...)``.

The load-step march advanced every buffered state on **every** cell: the readout formula ran whole-domain
because there was nowhere to say otherwise. That makes the ordinary multi-material problem inexpressible
— a plastic strip drawn over an elastic die, a damaging layer bonded to a sound one — because the strip's
return map would also be evaluated on the die's cells, with the strip's material constants, and written
into the die's buffer. Nothing reported it; the die simply carried a plastic strain it has no business
having, and whether that mattered depended on whether some other term happened to read it.

``region=`` masks the readout. Outside the region a state is **frozen**: its next value is the value it
already had, so a zero-initialised plastic strain leaves that region elastic for free, and no constitutive
formula is ever evaluated with the wrong material's constants.

The oracles here are independent statements of the same physics, not restatements of the output: a
region-restricted STATE must reproduce a hand-written region-restricted SOURCE, and J2 restricted to half
a bar must cap the stress there while the other half stays on the elastic line.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


NSTEP = 4


def _two_regions(tau=True):
    d = jno.Shape.regions(left=jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34),
                          right=jno.Shape.rect(1.0, 0.0, 2.0, 1.0, size=0.34),
                          conforming=False)
    d = d.domain(tau=(0.0, 1.0, NSTEP)) if tau else d.domain()
    _ = d.built_mesh
    return d


def _counter_fem(region):
    """Poisson driven by a state that counts load steps: ``s.evolves(s.i(-1) + 1)``.

    At step k the residual sees ``s.i(-1) = k - 1``, so the final step is a Poisson solve with a source
    of ``NSTEP - 1`` wherever the state was allowed to evolve, and 0 everywhere else.
    """
    n = jno.np
    d = _two_regions()
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(names=("s", "s_t"))
    ci, bd = d.variable("interior", split=True), d.variable("boundary", split=True)
    X = [ci[0], ci[1]]
    terms = [
        n.inner(n.grad(u, X), n.grad(phi, X), n_contract=1) - s.i(-1) * phi.bind(x=X[0], y=X[1]),
        s.evolves(s.i(-1) + 1.0) if region is None else s.evolves(s.i(-1) + 1.0, region=region),
        u(bd[0], bd[1]) - 0.0,
    ]
    return d, jno.fem(terms)


def _source_fem(where):
    """The SAME Poisson problem with the source written directly as a per-region term -- the independent
    statement the masked state has to reproduce. ``where`` is None (whole domain) or a region name."""
    n = jno.np
    d = _two_regions(tau=False)
    u, phi = d.fem_symbols()
    ci, bd = d.variable("interior", split=True), d.variable("boundary", split=True)
    X = [ci[0], ci[1]]
    src = float(NSTEP - 1)
    terms = [n.inner(n.grad(u, X), n.grad(phi, X), n_contract=1)]
    if where is None:
        terms.append(-src * phi.bind(x=X[0], y=X[1]))
    else:
        cw = d.variable(where, split=True)
        terms.append(-src * phi.bind(x=cw[0], y=cw[1]))
    terms.append(u(bd[0], bd[1]) - 0.0)
    return d, jno.fem(terms)


# ----------------------------------------------------------------------------------------------
# The oracle: a masked STATE must reproduce a hand-written masked SOURCE
# ----------------------------------------------------------------------------------------------
def test_a_state_restricted_to_a_region_matches_the_same_source_written_per_region():
    _, fem_state = _counter_fem("left")
    _, fem_src = _source_fem("left")
    marched = np.asarray(fem_state.solve())[-1].reshape(-1)
    direct = np.asarray(fem_src.solve()).reshape(-1)
    assert marched.shape == direct.shape
    err = np.abs(marched - direct).max() / max(float(np.abs(direct).max()), 1e-30)
    assert err < 1e-7, f"masked state and per-region source disagree by {err:.3e} relative"  # 1e-7: the BiCGStab floor


def test_the_unrestricted_state_still_matches_the_whole_domain_source():
    """The control for the test above: without `region=` nothing may change."""
    _, fem_state = _counter_fem(None)
    _, fem_src = _source_fem(None)
    marched = np.asarray(fem_state.solve())[-1].reshape(-1)
    direct = np.asarray(fem_src.solve()).reshape(-1)
    err = np.abs(marched - direct).max() / max(float(np.abs(direct).max()), 1e-30)
    assert err < 1e-7, f"unrestricted state moved: {err:.3e} relative"  # 1e-7: the BiCGStab floor


def test_restricting_to_one_region_is_not_the_same_as_the_whole_domain():
    """Guards against a `region=` that is accepted and then ignored -- the failure mode a passing
    equality test would hide."""
    _, fem_left = _counter_fem("left")
    _, fem_all = _counter_fem(None)
    a = np.asarray(fem_left.solve())[-1].reshape(-1)
    b = np.asarray(fem_all.solve())[-1].reshape(-1)
    rel = np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-30)
    assert rel > 0.05, f"restricted and whole-domain answers differ by only {rel:.3e}"


def test_an_unknown_region_is_refused_by_name():
    n = jno.np
    d = _two_regions()
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(names=("s", "s_t"))
    ci, bd = d.variable("interior", split=True), d.variable("boundary", split=True)
    X = [ci[0], ci[1]]
    with pytest.raises((ValueError, KeyError), match="middle"):
        jno.fem([
            n.inner(n.grad(u, X), n.grad(phi, X), n_contract=1) - s.i(-1) * phi.bind(x=X[0], y=X[1]),
            s.evolves(s.i(-1) + 1.0, region="middle"),
            u(bd[0], bd[1]) - 0.0,
        ]).solve()


# ----------------------------------------------------------------------------------------------
# The physical one: J2 in half a bar, the other half on the elastic line
# ----------------------------------------------------------------------------------------------
def test_plasticity_in_one_region_leaves_the_other_region_exactly_frozen():
    """J2 restricted to half a bar, asserted on the readout itself rather than on a downstream norm.

    Fed a NONZERO incoming history, the readout must return, on every cell outside the region, exactly
    the value that was already there — bit-identical, because the masking writes slot 0 back rather than
    recomputing something small. Inside the region it must actually move, or the mask would pass this
    test by doing nothing anywhere.
    """
    import jax.numpy as jnp

    from tests.test_fem_history_march import SY, _j2_stress

    n = jno.np
    d = jno.Shape.regions(left=jno.Shape.box(0, 0, 0, 1, 1, 1),
                          right=jno.Shape.box(1, 0, 0, 2, 1, 1),
                          conforming=False).domain(tau=(0.0, 1.0, 3))
    _ = d.built_mesh
    d.tag("lo", lambda x, y, z: x < 1e-9)
    d.tag("hi", lambda x, y, z: x > 2.0 - 1e-9)
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _e = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    al, _a = d.fem_symbols(value_shape=(), names=("al", "al_t"))
    ci = d.variable("interior", split=True)
    X = [ci[0], ci[1], ci[2]]
    cl, ch = d.variable("lo", split=True), d.variable("hi", split=True)
    sig, dg, nd = _j2_stress(u, X, ep.i(-1), al.i(-1), sy=SY)
    rt = float(np.sqrt(1.5))
    fem = jno.fem([
        n.inner(sig, n.sym(n.grad(phi, X)), n_contract=2),
        ep.evolves(ep.i(-1) + rt * dg * nd, region="left"),
        al.evolves(al.i(-1) + rt * dg, region="left"),
        u(cl[0], cl[1], cl[2])[0] - 0.0,
        u(cl[0], cl[1], cl[2])[1] - 0.0,
        u(cl[0], cl[1], cl[2])[2] - 0.0,
        u(ch[0], ch[1], ch[2])[0] - 0.05,
    ])
    op = fem.operator
    from jno.utils.solver.fem_utils import _cell_region_mask

    mask = np.asarray(_cell_region_mask(d, "left")).reshape(-1) > 0
    assert mask.any() and (~mask).any(), "the fixture must straddle both regions"

    rng = np.random.default_rng(0)
    bufs = {k: jnp.asarray(rng.normal(size=sp["shape"]) * 1e-3) for k, sp in op.history_specs.items()}
    uu = jnp.asarray(rng.normal(size=int(op.size)) * 1e-2)
    out = op.state_readout(uu, 1.0, {"__history__": bufs})

    for key, sp in op.history_specs.items():
        prev0 = np.asarray(bufs[key])[:, :, 0, ...]     # slot 0 IS this state's .i(-1)
        got = np.asarray(out[key])
        assert np.array_equal(got[~mask], prev0[~mask]), (
            f"state {sp['name']!r} was not frozen outside its region "
            f"(max change {np.abs(got[~mask] - prev0[~mask]).max():.3e})"
        )
        moved = np.abs(got[mask] - prev0[mask]).max()
        assert moved > 1e-9, f"state {sp['name']!r} did not advance inside its region either ({moved:.3e})"
