"""A linear transient march across several devices: the operator splits, the answer does not move.

The assembled ``M`` and ``A`` are partitioned on their nonzero axis and passed into the compiled scan as
arguments; the state stays replicated, so each step's Krylov solve is unchanged. Checked here on
simulated CPU devices (in a subprocess: ``XLA_FLAGS`` must precede JAX's start), against a one-device
march, and on the compiled program -- answers alone cannot show a partition that silently gathers the
operator back onto every device (``docs/fem/inverse.md`` records two such traps). No speed is asserted.
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
    import jax, numpy as np
    jax.config.update("jax_enable_x64", True)
    import jno
    import jno.utils.solver.backend_blocks as B

    hlo, placed = [], []
    real_jit, real_shard = jax.jit, None
    import jno.utils.solver.sharding as S
    real_shard = S.shard_triplets
    def shard_spy(data, idx, mesh):
        d, i = real_shard(data, idx, mesh)
        placed.append((int(d.shape[0]), int(d.addressable_shards[0].data.shape[0])))
        return d, i
    S.shard_triplets = shard_spy
    def jit_spy(fn=None, *args, **kw):
        if fn is None or getattr(fn, "__name__", "") != "march":  # everything else is jax's own jit
            return real_jit(fn, *args, **kw) if fn is not None else real_jit(*args, **kw)
        compiled = real_jit(fn, *args, **kw)
        def call(*a):
            hlo.append(compiled.lower(*a).compile().as_text())
            return compiled(*a)
        return call
    jax.jit = jit_spy

    d = jno.shape.rect(0, 0, 1, 1, size=0.06).domain(time=(0.0, 0.05, 11))
    u, v = d.fem_symbols()
    c = d.variable("interior", split=True); cb = d.variable("boundary", split=True); ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], t=c[2]), v.bind(x=c[0], y=c[1], t=c[2])
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(cb[0], cb[1]) - 0.0,
                   u(ci[0], ci[1]) - jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])])
    s = fem.solve(shard=None if len(sys.argv) < 2 else False)
    traj = np.asarray(s.fn() if hasattr(s, "fn") else s)
    text = hlo[0] if hlo else ""
    print("RESULT " + json.dumps({"traj": traj[-1].tolist(), "placed": placed,
          "all_reduce": "all-reduce" in text, "all_gather": "all-gather" in text, "compiled": bool(hlo)}))
    """
)


def _run(n_dev, opt_out=False):
    env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_dev}", "JAX_PLATFORMS": "cpu"}
    argv = [sys.executable, "-c", _INNER] + (["opt-out"] if opt_out else [])
    r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT "))[len("RESULT ") :])


@pytest.fixture(scope="module")
def one():
    return _run(1)


def test_one_device_takes_the_plain_march(one):
    assert not one["compiled"] and not one["placed"]


@pytest.mark.parametrize("n_dev", [2, 4])
def test_a_linear_march_splits_the_operator_and_keeps_the_answer(one, n_dev):
    r = _run(n_dev)
    assert r["compiled"], "the sharded march was not taken"
    for total, per_device in r["placed"]:  # M's and A's triplets
        assert per_device == total // n_dev, (total, per_device)
    assert r["all_reduce"] and not r["all_gather"], "the operator must never be gathered onto one device"
    np.testing.assert_allclose(r["traj"], one["traj"], rtol=0, atol=1e-13)


def test_shard_false_keeps_one_device():
    r = _run(4, opt_out=True)
    assert not r["compiled"] and not r["placed"]
