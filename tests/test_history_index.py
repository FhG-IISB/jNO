"""The step-time history primitive ``v.i(k)`` (:meth:`jno.trace.Placeholder.i`) and its keep-depth
inference (:func:`jno.trace.history_variables`).

``v.i(k)`` (``k <= 0``) is a variable read ``|k|`` load-steps back — the read side of path-dependent
physics (plastic history ``ep.i(-1)``) and multistep time schemes (``u.i(-2)``). The build infers how many
past states to buffer from the most-negative index each variable is used with. This file covers only that
trace-level primitive; the per-quadrature-point buffer threading + load-step driver are separate.
"""

from __future__ import annotations

import jax
import pytest

import jno
from jno.trace import HistoryRef, history_variables


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _domain():
    return jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.6).domain()


def test_i_builds_history_ref_carrying_base_shape_and_offset():
    d = _domain()
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    h = ep.i(-1)
    assert isinstance(h, HistoryRef)
    assert h.value_shape == (3, 3)  # inherits the base field's shape
    assert h.offset == -1


def test_i_forwards_through_scalar_and_matrix_views():
    """A state variable made with fem_symbols is a typed view; ``.i(k)`` must forward to the underlying
    field (so it works uniformly for a scalar α and a 3x3 tensor εp)."""
    d = _domain()
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    al, _ = d.fem_symbols(value_shape=(), names=("al", "al_t"))
    assert ep.i(-2).value_shape == (3, 3)
    assert al.i(-1).value_shape == ()


def test_positive_index_fails_loud():
    """``.i(k)`` is a PAST-state index; a future/positive offset is a mistake, not silently clamped."""
    d = _domain()
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    with pytest.raises(ValueError, match="k must be <= 0|PAST-state"):
        ep.i(1)


def test_keep_depth_inferred_from_most_negative_index():
    """A form reading ep at -1 and -2 and al at -1 must declare keep-depth {ep: 2, al: 1}."""
    d = _domain()
    co = d.variable("interior", split=True)
    coords = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    al, _ = d.fem_symbols(value_shape=(), names=("al", "al_t"))
    eu = jno.np.symgrad(u, coords)
    pe = jno.np.symgrad(phi, coords)
    sig = jno.np.function(lambda *a: a[0], [eu, ep.i(-1), ep.i(-2), al.i(-1)], name="stress")
    term = jno.np.inner(sig, pe, n_contract=2)
    hv = history_variables([term])
    depths = {getattr(b, "name", "?"): dep for (b, dep) in hv.values()}
    assert depths == {"ep": 2, "al": 1}


def test_offsets_of_one_base_share_a_buffer_key():
    """ep.i(-1) and ep.i(-2) must map to the SAME history buffer (one keep-depth-2 buffer, not two)."""
    d = _domain()
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    assert ep.i(-1).history_key == ep.i(-2).history_key


def test_current_step_needs_no_buffer():
    """Reading only ``.i(0)`` (the current step) declares no history — depth 0, nothing to buffer."""
    d = _domain()
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    assert history_variables([jno.np.function(lambda a: a, [ep.i(0)])]) == {}


def test_history_ref_is_not_the_unknown():
    """A HistoryRef is a KNOWN field (like a frozen field), so it must NOT register as a trial/unknown —
    that is what keeps a residual reading ep.i(-1) linear in the live unknown u."""
    from jno.trace import TrialFunction

    d = _domain()
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    assert not isinstance(ep.i(-1), TrialFunction)
