"""Data-parallel training over several devices: the work is split, and the answer does not move.

A PINN has one sample holding all its collocation points, so ``jno.core`` splits the POINTS over the
device mesh's batch axis (operator learning, with many samples, splits the samples). It used to tile the
single sample to one copy per device, so every device evaluated every point -- eight GPUs doing one
GPU's work (measured on 8 simulated devices: a full (1, 1, 4096, 2) on each).

Multi-device runs need ``XLA_FLAGS`` before JAX initialises, so the assertions run in a subprocess with
simulated CPU devices (the pattern of ``test_fem_sharding.py``). No speed is asserted -- the development
machine has one GPU; what is pinned is the split and that training gives the same network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_INNER = textwrap.dedent(
    """
    import json, sys
    import jax, numpy as np, foundax, optax
    jax.config.update("jax_enable_x64", True)
    import jno
    seen = {}
    orig = jno.core._shard_data
    def spy(self, data):
        out = orig(self, data)
        for k, v in out.items():
            if hasattr(v, "addressable_shards") and k not in seen:
                seen[k] = [list(v.shape), list(v.addressable_shards[0].data.shape)]
        return out
    jno.core._shard_data = spy
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    x, y, _ = d.variable("interior", sample=(512, None), split=True)
    net = jno.nn.wrap(foundax.mlp(2, 1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    u = (net(x, y) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)
    jno.core([(u.x.d(x) + u.y.d(y) + 1.0).mse], domain=d).solve(10)
    g = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)), -1).reshape(-1, 2)
    print("RESULT " + json.dumps({"placement": seen, "pred": np.asarray(net.module(jax.numpy.asarray(g))).ravel().tolist()}))
    """
)


def _run(n_dev):
    env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_dev}", "JAX_PLATFORMS": "cpu"}
    r = subprocess.run([sys.executable, "-c", _INNER], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


@pytest.fixture(scope="module")
def single():
    return _run(1)


@pytest.mark.parametrize("n_dev", [2, 8])
def test_a_pinn_splits_its_points_and_trains_the_same_network(single, n_dev):
    r = _run(n_dev)
    full, per_device = r["placement"]["interior"]
    assert full == [1, 1, 512, 2], full
    assert per_device == [1, 1, 512 // n_dev, 2], f"each device should hold 1/{n_dev} of the points, got {per_device}"
    np.testing.assert_allclose(r["pred"], single["pred"], rtol=0, atol=1e-12)
