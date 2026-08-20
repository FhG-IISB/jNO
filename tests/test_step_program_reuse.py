"""A second ``solve()`` reuses the compiled program instead of rebuilding it.

``jax.jit`` keys its trace and compilation cache on the wrapped function OBJECT, and the step
function is a fresh closure on every ``solve()``. So a chunked loop -- to checkpoint, to record a
diagnostic, to advance a curriculum -- re-traced, re-lowered and re-compiled a program it already
had. Measured on a 19,462-tet topology optimisation: 3.7 s of XLA per call for the hook gradient
alone, and 100 iterations in chunks of 10 took 3.1 min against 1.7 min in one call.

The reuse has to be conditional, which is what these tests pin: the same configuration reuses, a
changed one rebuilds. Getting that backwards is worse than not caching, because a stale executable
computes the wrong thing silently.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


def _core(n=1, lr=1e-2):
    p = jno.np.parameter((n,), name="p")
    p.initialize(lambda k, sh, dtype=None: jnp.zeros(sh))
    p.optimizer(optax.sgd(lr))
    return p, jno.core([((p[0] - 1.0) ** 2).name("obj")])


class TestStepProgramReuse:
    def test_a_second_solve_reuses_the_compiled_step(self):
        """The cache is populated on the first solve and hit on the second."""
        _, crux = _core()
        crux.solve(2)
        cache = getattr(crux, "_jit_program_cache", None)
        assert cache is not None and "step" in cache, "the first solve must populate the cache"
        first = cache["step"]
        crux.solve(2)
        assert crux._jit_program_cache["step"] is first, "the second solve must reuse the same jitted callable"

    def test_reuse_does_not_change_the_answer(self):
        """Two chunks and one chunk of the same length must land in the same place.

        With optimizer state carried across calls (``_optimizer_states_match``) the two are the same
        optimisation, so reusing the program must not perturb it.
        """
        p1, c1 = _core()
        c1.solve(20)
        a = float(np.asarray(c1.eval([p1])).reshape(-1)[0])
        p2, c2 = _core()
        c2.solve(10)
        c2.solve(10)
        b = float(np.asarray(c2.eval([p2])).reshape(-1)[0])
        assert a == pytest.approx(b, rel=1e-9), f"chunking changed the result: {a} vs {b}"

    def test_a_different_parameter_shape_rebuilds(self):
        """The key carries shapes, so a differently-sized problem cannot collide with a cached one.

        Separate cores, but the check that matters is that the KEY differs -- a shared key would mean
        a program traced for one shape being handed arrays of another.
        """
        _, c1 = _core(n=1)
        c1.solve(2)
        _, c2 = _core(n=4)
        c2.solve(2)
        assert c1._jit_program_cache["__key__"] != c2._jit_program_cache["__key__"]

    def test_a_swapped_optimizer_type_rebuilds(self):
        """Optimizers enter the key by type name, so exchanging one invalidates the cache."""
        p = jno.np.parameter((1,), name="p")
        p.initialize(lambda k, sh, dtype=None: jnp.zeros(sh))
        p.optimizer(optax.sgd(1e-2))
        c1 = jno.core([((p[0] - 1.0) ** 2)])
        c1.solve(2)
        k1 = c1._jit_program_cache["__key__"]

        p.optimizer(optax.adam(1e-2))
        c2 = jno.core([((p[0] - 1.0) ** 2)])
        c2.solve(2)
        assert c2._jit_program_cache["__key__"] != k1, "a different optimizer must not reuse the program"

    def test_an_undescribable_configuration_simply_does_not_cache(self):
        """``None`` never compares equal to a real key, so a configuration that cannot be described
        rebuilds rather than reusing something it should not."""
        _, crux = _core()
        assert crux._step_program_key(object(), None, None, "not a mapping", ()) is None
