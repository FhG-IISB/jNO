"""Partitioning an assembled FEM operator across devices.

The claim under test is that the **assembled BCOO needs no change of representation** to be
parallelised — its ``nnz`` axis partitions, each device scatter-adds its slice, and the partials are
combined by one ``all-reduce``. Consequently *no solver code changes*: ``cg``, ``bicgstab``,
``gmres`` and jNO's own ``minres``/``fgmres`` all run unmodified, Jacobi included.

Multi-device behaviour cannot be tested in-process: ``XLA_FLAGS`` is read when the JAX backend
initialises, and ``tests/conftest.py`` imports jax before any test module runs. So the real
assertions live in ``tests/_sharding_inner.py`` and are executed in a subprocess with a simulated
device count — the same subprocess-with-``env`` pattern ``test_tutorial_examples.py`` uses.

**No performance is asserted.** This machine has one GPU; a multi-device speedup is not measurable
here and must not be claimed. What *is* verifiable — and is what these tests pin — is correctness,
the even split, and the choice of collective.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_INNER = str(Path(__file__).parent / "_sharding_inner.py")


def _run(n_dev: int):
    """Run the inner assertions with ``n_dev`` simulated CPU devices."""
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_dev}",
        "JAX_PLATFORMS": "cpu",
    }
    return subprocess.run([sys.executable, _INNER, str(n_dev)], env=env, capture_output=True, text=True, timeout=900)


# Deliberately NOT marked slow. The CI runner deselects `-m 'not slow'`, and a slow-marked test is
# invisible to it — which is how periodic-Morley stayed broken on main unnoticed. The whole file runs
# in ~20s even paying three interpreter starts, so the marker would buy nothing and cost the coverage.
@pytest.mark.parametrize("n_dev", [1, 2, 4])
def test_sharded_operator_matches_single_device(n_dev):
    """Correctness, the even split, and the collective — across 1, 2 and 4 simulated devices.

    ``n_dev=1`` matters as much as the others: it is the degenerate partition, and it must behave
    exactly like today rather than take a different path."""
    r = _run(n_dev)
    assert r.returncode == 0, f"n_dev={n_dev} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    assert f"OK n_devices={n_dev}" in r.stdout, r.stdout[-2000:]


def test_padding_is_exact_and_only_when_needed():
    """The triplet count must divide the device count or ``device_put`` raises ``IndivisibleError``.
    Padding appends zero-valued triplets at ``(0, 0)``, which scatter-add to nothing — safe by
    construction rather than by masking, so nothing downstream needs a special case."""
    import jax.numpy as jnp

    from jno.utils.solver.sharding import pad_triplets

    data = jnp.arange(10.0)
    idx = jnp.stack([jnp.arange(10, dtype=jnp.int32), jnp.arange(10, dtype=jnp.int32)], axis=1)

    d, i, n_pad = pad_triplets(data, idx, 4)
    assert n_pad == 2 and d.shape[0] == 12 and i.shape == (12, 2)
    assert float(jnp.sum(d[10:])) == 0.0, "padded values must be zero"
    assert int(jnp.sum(i[10:])) == 0, "padded indices must point at (0, 0)"
    assert np.allclose(np.asarray(d[:10]), np.arange(10.0)), "the real triplets must be untouched"

    d2, i2, n2 = pad_triplets(data, idx, 5)
    assert n2 == 0 and d2.shape[0] == 10, "an already-divisible count must not be padded"


def test_device_request_is_explicit_not_silently_expanded():
    """``jno.core`` expands a default ``mesh=(1,1)`` to *all* visible devices. A FEM operator must
    not: silently distributing an operator nobody asked to distribute is a behaviour change, and the
    whole value of the default is that it changes nothing. An over-request fails loud instead."""
    import jax

    from jno.utils.solver.sharding import resolve_devices

    assert resolve_devices(None) == [], "the default must request no devices at all"
    assert len(resolve_devices(1)) == 1

    with pytest.raises(ValueError, match="positive device count"):
        resolve_devices(0)
    with pytest.raises(ValueError, match="only .* device"):
        resolve_devices(len(jax.devices()) + 8)
    with pytest.raises(ValueError, match="empty device list"):
        resolve_devices([])
