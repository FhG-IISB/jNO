"""``crux.eval(..., domain=other)`` fills the tags ``other`` lacks from the training domain.

It must not copy them: a per-sample training tag (labels of B samples) was duplicated into every eval domain with
``np.asarray(value).copy()``, one full copy of the training data per eval domain, then moved to the device with the
rest of that domain's context. A per-sample tag whose sample count differs from the eval domain's cannot be paired
with its samples at all, so it is not carried over; a tag that can be (one shared sample, or the same count) is
shared read-only.
"""

import numpy as np
import optax

import jno

N = 6


def _crux(n_train):
    import foundax
    import jax

    d = n_train * jno.domain(constructor=jno.domain.line(mesh_size=0.25))
    x, *_ = d.variable("interior")
    d.variable("p", np.arange(n_train, dtype=np.float32).reshape(n_train, 1, 1))
    d.variable("lab", np.ones((n_train, 1, N, 1), np.float32))
    d.variable("shared", np.full((1, 1, 1), 2.0, np.float32))
    net = jno.nn(foundax.mlp(1, output_dim=1, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    return jno.core([(net(x) - 0.0).mse]), d, net


def _eval_domain(n):
    d = n * jno.domain(constructor=jno.domain.line(mesh_size=0.25))
    x, *_ = d.variable("interior")
    return d, x


def test_per_sample_training_tags_are_not_copied_into_an_eval_domain_of_another_size():
    crux, train, net = _crux(8)
    d, x = _eval_domain(3)
    crux.eval([net(x)], domain=d)
    assert "lab" not in d.context and "p" not in d.context


def test_tags_that_fit_the_eval_domain_are_shared_not_copied():
    crux, train, net = _crux(8)
    d, x = _eval_domain(3)
    crux.eval([net(x)], domain=d)
    assert "shared" in d.context
    assert np.shares_memory(np.asarray(d.context["shared"]), np.asarray(train.context["shared"]))
    assert not np.asarray(d.context["shared"]).flags.writeable
