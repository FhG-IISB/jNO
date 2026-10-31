"""What a design loop on the GPU sends across the bus. Most GPUs are memory bound, and a table that is on the card already
should not be read back, copied again or worked on on the host.

- A compiled program takes its constants as arguments, one buffer per content however many programs read it (oracle: the
  device addresses of the live arrays), and two arrays are only one buffer when their content is equal (oracle: the
  programs' values).
- One step of a constrained loop reads what MMA needs from the card (design, gradient, Jacobian: its size is known from
  the problem) and scalars, and sends back the step: the operator never crosses.
  Oracle: JAX's own transfer log, read in a process of its own.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")

import jno.utils.solver.cudss_sym as cs  # noqa: E402

pytestmark = pytest.mark.skipif(not cs.available(), reason="needs a GPU with the cuDSS extension")

_SHARED = """
import numpy as np, jax, jax.numpy as jnp, jno, json
device = jax.devices()[0]
cpu = jax.devices("cpu")[0]


def buffers(shape):
    return len({x.unsafe_buffer_pointer() for x in jax.live_arrays() if x.shape == shape})


out = {}
with jax.default_device(cpu):
    host = jnp.arange(50000.0) + 0.5  # how a table leaves assembly: an array on the host's JAX device
programs = [jax.jit(lambda t, k=k: jnp.sum(t * host + k)) for k in range(3)]
values = [float(f(jnp.asarray(1.0))) for f in programs]
out["values"] = values
out["expected"] = [float(np.sum(np.arange(50000.0) + 0.5) + 50000 * k) for k in range(3)]
out["buffers_while_held"] = buffers((50000,))
del programs
import gc

jax.clear_caches()
gc.collect()
out["buffers_after"] = buffers((50000,))

# the same content in another array is one buffer; other content is another
with jax.default_device(cpu):
    twin = jnp.arange(50000.0) + 0.5
    other = jnp.arange(50000.0) + 1.5
held = [jax.jit(lambda t: jnp.sum(t * host)), jax.jit(lambda t: jnp.sum(t - twin)), jax.jit(lambda t: jnp.sum(t + other))]
out["twin_values"] = [float(f(jnp.asarray(1.0))) for f in held]
out["twin_buffers"] = buffers((50000,))

# numpy arrays are told apart by content too
first_np = np.arange(40000.0)
second_np = np.arange(40000.0) + 1.0
twin_np = np.arange(40000.0)
kept = [jax.jit(lambda t, c=c: jnp.sum(t * c)) for c in (first_np, second_np, twin_np)]
out["numpy_values"] = [float(f(jnp.asarray(1.0))) for f in kept]
out["numpy_expected"] = [float(np.sum(c.astype(np.float32))) for c in (first_np, second_np, twin_np)]
out["numpy_buffers"] = buffers((40000,))
# a program with a host callback skips JAX's fast path, and is asked for its constants at every call
import hashlib

hashed, real_sha1 = [], hashlib.sha1
hashlib.sha1 = lambda *a, **k: hashed.append(1) or real_sha1(*a, **k)
with jax.default_device(cpu):
    table = jnp.arange(30000.0)
called = []
noisy = jax.jit(lambda t: (jax.debug.callback(lambda: called.append(1)), jnp.sum(t * table))[1])
out["noisy"] = [float(noisy(jnp.asarray(1.0))) for _ in range(4)]
jax.effects_barrier()
hashlib.sha1 = real_sha1
out["noisy_called"], out["noisy_hashed"] = len(called), len(hashed)
print("RESULT " + json.dumps(out))
"""

_LOOP = """
import os
import numpy as np, jax, jax.numpy as jnp, jno, json
from jno.utils.adaptive.callbacks import Callback

jax.config.update("jax_enable_x64", True)
inner = jno.np.inner
d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.02).domain()
x, y = d.variable("interior", split=True)[:2]
xb, yb = d.variable("boundary", split=True)[:2]
g = lambda f: jno.np.jacobian(f, [x, y])
nc = len(d.mesh.cells_dict["triangle"])
_r, t0 = d.fem_symbols(space="P0", names=("r", "t0"))
rho = jno.np.parameter(t0, name="rho")
rho.dtype(jnp.float64)
rho.initialize(jax.nn.initializers.constant(0.5))
rho.optimizer(jno.optimizers.mma(move=0.05, lower=0.1, upper=1.0))
a, v = d.fem_symbols(names=("a", "v"))
A, V = a.bind(x=x, y=y), v.bind(x=x, y=y)
field = jno.fem([rho * inner(g(A), g(V)) - V, a(xb, yb) - 0.0])
U = A.freeze(field.solve(linear=jno.solve.lu(backend="cudss")))
energy = (U * 1.0).integrate(d)
volume = (rho * 1.0).integrate(d)
crux = jno.core([energy, jno.le(volume, 0.6)], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))


class Mark(Callback):
    n = 0

    def on_epoch_end(self, **kw):
        Mark.n += 1
        if Mark.n == 3:
            os.write(2, b"MARK start\\n")
        if Mark.n == 8:
            os.write(2, b"MARK stop\\n")
        return False


jax.config.update("jax_transfer_guard", "log_explicit")
crux.solve(10, callbacks=[Mark()])
jax.config.update("jax_transfer_guard", "allow")
print("RESULT " + json.dumps({"nnz": int(field._linear_solver_context()["pattern"].shape[0]), "design": nc}))
"""


def _run(code, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "JNO_HOIST_CONSTANTS")}
    env.update({"TF_CPP_MIN_LOG_LEVEL": "0", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.2", "JAX_PLATFORMS": "cuda,cpu", **extra})
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(line[-1][7:]), r.stderr


def test_programs_share_one_buffer_per_constant():
    out, _log = _run(_SHARED)
    np.testing.assert_allclose(out["values"], out["expected"], rtol=1e-6)
    assert out["buffers_while_held"] == 1, "a host array stood on the card once per program"
    assert out["buffers_after"] == 0, "the shared buffer outlived the programs that held it"
    assert out["twin_buffers"] == 2, "equal content is one buffer, other content another"
    np.testing.assert_allclose(out["numpy_values"], out["numpy_expected"], rtol=1e-3)
    assert out["numpy_values"][0] != out["numpy_values"][1], "other content read as the first array"
    assert out["numpy_buffers"] == 2, "equal numpy content is one buffer, other content another"
    assert out["noisy_called"] == 4, "the callback did not run: the test sees nothing"
    assert len(set(out["noisy"])) == 1
    assert out["noisy_hashed"] <= 2, f"a constant was hashed again at each call: {out['noisy_hashed']}"


_SHAPE = re.compile(r"(?:aval=ShapedArray\(\w+\[([\d,\s]*)\]|shape=\(([\d,\s]*)\))")


def _size(line):
    hit = _SHAPE.search(line)
    if hit is None:
        return 0
    n = 1
    for part in (hit.group(1) or hit.group(2) or "").replace(" ", "").split(","):
        if part:
            n *= int(part)
    return n


def test_a_step_reads_what_the_optimiser_needs_and_the_operator_never_crosses():
    out, log = _run(_LOOP)
    start, stop = log.index("MARK start"), log.index("MARK stop")
    lines = [
        ln for ln in log[start:stop].splitlines() if "transfer:" in ln and "CpuDevice" not in ln.split("dst_sharding")[-1]
    ]
    assert lines, "no transfer logged: the test sees nothing"
    biggest = max(_size(ln) for ln in lines)
    design = out["design"]
    # the optimiser reads the design, the gradient and one row of the Jacobian per constraint, each `design` long
    assert biggest <= design, f"a step moved {biggest} entries; the design has {design}"
    assert biggest < out["nnz"], f"a step moved {biggest} entries; the operator has {out['nnz']}"
    per_step = sum(_size(ln) for ln in lines) / 5
    assert per_step <= 6 * design, f"{per_step} entries per step for a design of {design}"


_UPLOAD = """
import numpy as np, jax, jax.numpy as jnp, jno, json
from jno.utils.solver.placement import upload, _UPLOAD_CHUNK_BYTES

jax.config.update("jax_enable_x64", True)
out = {"chunk": _UPLOAD_CHUNK_BYTES}
rng = np.random.default_rng(0)
stats = lambda: jax.local_devices()[0].memory_stats()
for name, a in (("small", rng.normal(size=1000)), ("large", rng.normal(size=(50_000_000, 2)))):
    y = upload(a)
    out[name] = {"equal": bool(np.array_equal(np.asarray(y), a)), "committed": bool(y.committed), "dtype": str(y.dtype),
                 "same_shape": tuple(y.shape) == a.shape, "bytes": a.nbytes}
    del y
idx = upload(rng.integers(0, 1000, (40_000_000, 2), dtype=np.int32))
out["int_equal"] = idx.dtype == jnp.int32
out["peak"] = stats()["peak_bytes_in_use"]
print("RESULT " + json.dumps(out))
"""


def test_a_large_array_reaches_the_card_in_little_more_than_its_size():
    """a plain transfer needs twice the array: its staging buffer is as large as it is. Oracle: the array read back"""
    out, _log = _run(_UPLOAD, XLA_PYTHON_CLIENT_MEM_FRACTION="0.5")
    for name in ("small", "large"):
        assert out[name]["equal"] and out[name]["same_shape"], name
        assert not out[name]["committed"], "the placement stays implicit"
        assert out[name]["dtype"] == "float64"
    assert out["int_equal"]
    # 800 MB and then 320 MB: without chunks the 800 MB array alone needs 1.9 GB
    assert out["peak"] < out["large"]["bytes"] + 3.2 * out["chunk"], out["peak"]
