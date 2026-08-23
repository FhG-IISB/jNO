"""A ``constrain()`` transform's scalar constants must reach the step-program key.

A reparameterization lives in the STATIC half of the partition, so a Python value its transform
closes over is a trace-time constant. That is the ordinary ``jax.jit`` contract and it is fine --
right up until a continuation schedule mutates one between ``solve()`` calls, which is what every
density-projection ramp does (``beta = [1.0]``, then ``beta[0] = 2.0``). Nothing else in the key
moves when it does: not the tree, not a shape, not a dtype. So the cached executable is reused, the
gradient is taken at the OLD value, and ``crux.eval`` -- which recompiles -- reports the NEW one.

Right-looking numbers, wrong optimisation, nothing raised. The run logs a rising beta and a falling
grey-level indicator while the design is optimised at the beta it started with.

What is deliberately NOT asserted here: any particular compile count, and any claim that the key
catches every mutation. It catches SCALARS in the transform's closure cells and in the globals its
code references. A mutated numpy array and a mutated attribute on an object still slip through --
see :func:`jno.core._reparam_scalar_constants` for why that boundary is where it is.
"""

import sys

import jax
import numpy as np
import optax
import pytest

import jno

# Module scope on purpose. `def physical(r)` written at module level reads its beta cell as a
# GLOBAL, not as a closure cell, and every problem script in the wild is written that way. A test
# that declares it inside the test function exercises the closure route only and would pass against
# a fix that reads `__closure__` alone -- which was the first attempt, and it did not work.
_BOX = [1.0]


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _scaled_parameter():
    """One parameter, physical = _BOX[0] * design, loss = (physical - 1)^2, plain SGD at lr 1.

    Deliberately the smallest thing that can show the bug: with ``sgd(1.0)`` the step IS the
    gradient, so the stored value after one solve is a closed form and the assertion can name both
    the stale and the live answer rather than comparing against a tolerance.
    """
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    p = jno.np.parameter((1,), name="p")
    p.dtype(jax.numpy.float64)
    p.initialize(lambda k, sh, dtype=None: jax.numpy.full(sh, 0.5))
    p.constrain(lambda r: _BOX[0] * r)
    p.optimizer(optax.sgd(1.0))
    crux = jno.core([((p[0] - 1.0) ** 2).name("L")], domain=d)
    return crux, p


def _stored(crux, p):
    """The DESIGN value, undoing the transform ``crux.eval`` applies."""
    return float(np.asarray(crux.eval([p])).reshape(-1)[0]) / _BOX[0]


class TestMutatedReparamConstant:
    """Mutating a scalar a ``constrain()`` transform reads must change the next step."""

    def test_the_mutation_reaches_the_gradient(self):
        _BOX[0] = 1.0
        crux, p = _scaled_parameter()

        crux.solve(1)  # loss = (r - 1)^2 at r = 0.5 -> grad 2(0.5-1) = -1 -> r = 1.5
        assert _stored(crux, p) == pytest.approx(1.5, abs=1e-9), (
            "the box=1 step is a closed form; if this fails the harness is wrong, not the cache"
        )

        _BOX[0] = 10.0
        crux.solve(1)  # loss = (10r - 1)^2 at r = 1.5 -> grad 20(15-1) = 280 -> r = -278.5
        got = _stored(crux, p)
        assert got == pytest.approx(-278.5, rel=1e-9), (
            f"the mutated constant did not reach the gradient: stored p = {got}, but the live "
            f"box=10 program gives -278.5 and the stale box=1 program gives 0.5. Getting 0.5 here "
            f"means the step program was reused across a changed reparameterization."
        )

    def test_an_unchanged_constant_still_reuses_the_program(self):
        """The fix must not defeat the cache it is narrowing.

        `d8f5cf6` reuses the compiled step across `solve()` calls and measured 3.1 min -> 1.7 min on
        a chunked topology-optimisation run. A key that changed every call would give that back.
        """
        _BOX[0] = 1.0
        crux, _p = _scaled_parameter()
        keys = []
        original = crux._step_program_key

        def _spy(*a, **k):
            key = original(*a, **k)
            keys.append(key)
            return key

        crux._step_program_key = _spy
        crux.solve(1)
        crux.solve(1)
        crux._step_program_key = original

        assert len(keys) >= 2, "the key must be built on every solve, or reuse cannot be decided"
        assert keys[0] is not None, "a describable configuration must produce a key, not None"
        assert keys[0] == keys[1], (
            "nothing changed between these two solves, so the key must be identical and the "
            "compiled program reused"
        )

    def test_the_key_moves_when_the_constant_moves(self):
        """The narrow claim, asserted directly on the key rather than through a solve."""
        _BOX[0] = 1.0
        crux, _p = _scaled_parameter()
        crux.solve(1)
        # `jno.core` is the class, so reach the module it lives in rather than guessing a path.
        _mod = sys.modules[type(crux).__module__]
        before = _mod._reparam_scalar_constants(crux.models)
        _BOX[0] = 10.0
        after = _mod._reparam_scalar_constants(crux.models)
        assert before != after, (
            f"the reparameterization constants must differ once the cell is mutated; got {before} "
            f"both times, so the key cannot tell the two programs apart"
        )
        assert 1.0 in before and 10.0 in after, (
            f"the mutated value itself must appear: before={before}, after={after}"
        )
