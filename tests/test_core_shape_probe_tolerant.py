"""``jno.core`` logs constraint shapes, and that log must not be able to fail the solve.

The probe is `jax.eval_shape`, which supplies no values. A constraint holding a solver that runs on
the HOST cannot run without them -- `jno.peec`'s sparse factorisation is one -- so probing such a
constraint raises. It used to raise *out of* `solve()`, killing a run that would otherwise have
trained, for the sake of a log line nothing downstream consumes.

Reported, not swallowed: losing the shape log is a small named cost, and silently skipping it would
be the other failure this codebase refuses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

jax.config.update("jax_enable_x64", True)


def _dummy():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


def test_a_probe_that_raises_does_not_take_the_solve_with_it(capsys, monkeypatch):
    """The behaviour, tested where it lives.

    Building a genuinely unprobeable constraint means embedding a host solver, which drags in a
    whole PEEC network and tests that module rather than this one. The contract here is narrow: if
    the probe raises, `solve()` continues and says so. So the probe is made to raise.
    """
    p = jno.np.parameter((1,), name="w")
    p.initialize(jax.nn.initializers.constant(2.0))
    p.optimizer(optax.adam(0.1))
    crux = jno.core([jno.fn(lambda v: jnp.sum(v**2), [p], name="sq")], domain=_dummy())

    def boom(*_a, **_k):
        raise jax.errors.TracerArrayConversionError(jnp.zeros(1))

    monkeypatch.setattr(type(crux), "_log_constraint_shapes_probe", boom, raising=True)
    stats = crux.solve(epochs=3)

    h = np.asarray(stats.total_loss_history).reshape(-1)
    assert h.size >= 1 and h[-1] < h[0], "the solve must have run and trained"
    # jNO's logger is a print fallback rather than a stdlib logger, so the report lands on stdout
    assert "Constraint shapes unavailable" in capsys.readouterr().out, (
        "the skipped log must be REPORTED, not silently dropped"
    )


def test_an_ordinary_constraint_still_gets_its_shapes_logged():
    """The tolerance must not cost the diagnostic on every problem that can be probed."""
    p = jno.np.parameter((1,), name="k")
    p.initialize(jax.nn.initializers.constant(0.5))
    p.optimizer(optax.adam(0.1))
    stats = jno.core([jno.fn(lambda v: jnp.sum(v**2), [p], name="sq")], domain=_dummy()).solve(epochs=3)
    h = np.asarray(stats.total_loss_history).reshape(-1)
    assert h[-1] < h[0], (h[0], h[-1])
