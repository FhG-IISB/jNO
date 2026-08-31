"""The preconditioner's sparse LU is not rebuilt for values it has already seen.

Profiled on a GPU at 40,000 nodes, ``_superlu.gstrf`` was 1.515 s per solve -- 53 % of the whole
thing -- and its fill-in grows faster than the device work it accelerates (21x the nonzeros already
at 8,640 nodes). Re-running it for the same conductivity and the same frequency buys nothing, and it
was being re-run on every ``solve()``.

What the cache does NOT claim is that a stale factorisation still accelerates a CHANGED conductivity.
It is keyed on content, so a design loop that moves sigma every iteration misses it by construction,
and that is the honest behaviour: the alternative is a different claim about preconditioning, not a
caching decision.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver import peec as P

jax.config.update("jax_enable_x64", True)

SIG, LX, WY, TZ = 5.8e7, 0.040, 0.004, 0.002


def _counted():
    """Wrap scipy's factorisation so a test can count how often it actually runs."""
    import scipy.sparse.linalg as spla

    calls = {"n": 0}
    real = spla.splu

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    return calls, real, counting


def _built(pitch=0.002, freq=0.0):
    bar = jno.Shape.box(0, 0, 0, LX, WY, TZ, size=pitch).attach(sigma=SIG).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < pitch)
    d.tag("B", lambda x, y, z: x > LX - pitch)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build()


def test_repeating_the_same_solve_does_not_refactorise():
    import scipy.sparse.linalg as spla

    b = _built()
    b.solve().R.block_until_ready()  # warm compile and prime the holder
    calls, real, counting = _counted()
    spla.splu = counting
    try:
        for _ in range(4):
            b.solve().R.block_until_ready()
    finally:
        spla.splu = real
    assert calls["n"] == 0, f"{calls['n']} factorisations for four identical solves"


def test_a_changed_conductivity_DOES_refactorise():
    """The other half of the contract: the cache must not serve a matrix it no longer describes."""
    import scipy.sparse.linalg as spla

    b = _built()
    b.solve(sigma={"bar": SIG}).R.block_until_ready()
    calls, real, counting = _counted()
    spla.splu = counting
    try:
        b.solve(sigma={"bar": 0.5 * SIG}).R.block_until_ready()
    finally:
        spla.splu = real
    assert calls["n"] >= 1, "a different conductivity reused a factorisation built for another one"


def test_the_answer_is_unchanged_by_caching():
    """A cached preconditioner must not move the result -- it only accelerates."""
    b = _built()
    first = complex(b.solve().R)
    again = complex(b.solve().R)
    assert abs(again - first) <= 1e-12 * max(abs(first), 1.0)
    # and a genuinely different conductivity still gives the right answer, not the cached one
    half = complex(b.solve(sigma={"bar": 0.5 * SIG}).R)
    assert abs(half / first - 2.0) < 1e-9  # R goes as 1/sigma, exactly


def test_a_frequency_sweep_refactorises_per_frequency():
    """The Schur complement carries jw, so two frequencies are two different matrices."""
    import scipy.sparse.linalg as spla

    b = _built(freq=0.0)
    b.solve().R.block_until_ready()
    b2 = _built(pitch=0.002, freq=1e4)  # one cell through the thickness: legal at any frequency
    b2.solve().R.block_until_ready()
    calls, real, counting = _counted()
    spla.splu = counting
    try:
        b.solve().R.block_until_ready()  # back to the DC network: its key is no longer the holder's
    finally:
        spla.splu = real
    assert calls["n"] >= 1, "a one-entry cache served a matrix built at another frequency"


def test_a_failed_factorisation_leaves_no_key_claiming_success():
    """If splu raises, the holder must not keep a key that would skip the rebuild next time."""
    import scipy.sparse.linalg as spla

    b = _built()
    b.solve().R.block_until_ready()
    real = spla.splu
    spla.splu = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with pytest.raises(Exception):
            b.solve(sigma={"bar": 0.25 * SIG}).R.block_until_ready()
    finally:
        spla.splu = real
    assert P._LU_HOLDER.get("key") is None
    # and the next real solve still works rather than reusing a corpse
    assert np.isfinite(float(np.real(b.solve(sigma={"bar": 0.25 * SIG}).R)))
