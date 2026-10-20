"""Live progress from inside compiled solves, without stopping them: the ``log=`` of every ``.solve``.

A compiled loop cannot print without a host callback, and a host callback makes the device wait for Python
every time it fires. Here the loop writes a few scalars into a ring buffer in host memory instead
(``jno/_ext/beacon_ffi.cu``): on the GPU a one-thread kernel writes them through mapped pinned memory and the
stream goes on, on the CPU the handler is a few stores. A reader thread prints them::

    fem.solve(log=True)      # newton step 1: residual 3.1e-02 ...  as the steps happen
    fem.solve(log=10)        # every 10th step of each loop, and its last

A solve reports only when it is given ``log=``, so a solve inside another one (a ``jno.fem`` in a
``jno.core`` loss) stays quiet unless it asks. Without ``log=`` nothing is traced: :func:`emit` is not called,
and the program is the one jNO built before this existed. Whether a program reports is decided when it is
traced, and jNO's compiled-solve caches key on :func:`settings`.

The values are not differentiated (``stop_gradient``), and under ``vmap`` each batch member emits in turn.
A rematerialised step (``jax.checkpoint`` in a reverse-mode march) emits again when the backward pass
recomputes it. When more than 4096 lines land between two reads, the oldest are overwritten and the reader
says how many it missed.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "_ext" / "beacon_ffi.cu"
_ACTIVE = [False]
_EVERY = [None]  # report every N-th step of a loop (and its last); None: every step
_POLL = 0.1  # seconds between two reads of the rings
_READER = [None]
_READER_LOCK = threading.Lock()
_LIB: dict = {}
_TAGS: dict[str, int] = {}
_FORMATS: dict[int, tuple[str, str]] = {}


def register(name: str, fmt: str) -> int:
    """Name a beacon and the line it prints (``str.format`` over its values); returns its tag."""
    if name not in _TAGS:
        _TAGS[name] = len(_TAGS) + 1
    _FORMATS[_TAGS[name]] = (name, fmt)
    return _TAGS[name]


register("newton", "step {0:.0f}: residual {1:.3e} (target {2:.1e}), forcing {3:.1e}")
register("time", "step {0:.0f}: t = {1:.6g}, max |u| = {2:.4g}")
register("eigs", "iteration {0:.0f}: residual {1:.3e} (target {2:.1e})")


def active() -> bool:
    """Whether the solve being traced asked for ``log=``: callers trace :func:`emit` only then."""
    return _ACTIVE[0]


def settings() -> tuple:
    """What a traced program bakes in from ``log=``: part of every compiled-solve cache key."""
    return (_ACTIVE[0], _EVERY[0])


def every() -> int | None:
    """The ``N`` of ``log=N`` for the solve being traced (None: every step)."""
    return _EVERY[0]


def _build(gpu: bool) -> Path:
    import jax

    inc = Path(jax.__file__).resolve().parents[1] / "jaxlib" / "include"
    key = hashlib.sha256(_SRC.read_bytes() + str(inc).encode() + bytes([gpu])).hexdigest()[:16]
    cache = Path(os.environ.get("JNO_EXT_CACHE", Path.home() / ".cache" / "jno" / "ext"))
    so = cache / f"libjno_beacon_{'gpu' if gpu else 'cpu'}_{key}.so"
    if so.exists():
        return so
    cache.mkdir(parents=True, exist_ok=True)
    tmp = so.with_suffix(f".{os.getpid()}.tmp")
    if gpu:
        from .solver.cudss_sym import _nvcc

        nvcc = _nvcc()
        if nvcc is None:
            raise RuntimeError("jno.utils.beacon: building the GPU beacon needs nvcc on PATH (or NVCC=...).")
        cmd = [nvcc, "-O2", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-I{inc}", str(_SRC), "-o", str(tmp)]
    else:
        cmd = [os.environ.get("CXX", "c++"), "-O2", "-std=c++17", "-shared", "-fPIC", "-x", "c++", f"-I{inc}",
               str(_SRC), "-o", str(tmp)]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("jno.utils.beacon: compiling the beacon failed:\n" + r.stderr[-3000:])
    tmp.replace(so)
    return so


def _load(gpu: bool):
    """The extension for this platform, built on first use and its FFI targets registered."""
    if gpu in _LIB:
        return _LIB[gpu]
    import jax

    lib = ctypes.cdll.LoadLibrary(str(_build(gpu)))
    lib.jno_beacon_ring.restype = ctypes.c_void_p
    lib.jno_beacon_ring.argtypes = [ctypes.c_int]
    lib.jno_beacon_slots.restype = ctypes.c_int64
    lib.jno_beacon_values.restype = ctypes.c_int64
    plat, kind = ("CUDA", "gpu") if gpu else ("cpu", "cpu")
    for dt in ("f64", "f32"):
        name = f"jno_beacon_emit_{kind}_{dt}"
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, name)), platform=plat)
    if (lib.jno_beacon_init_gpu() if gpu else lib.jno_beacon_init_cpu()) != 0:
        raise RuntimeError("jno.utils.beacon: could not allocate the ring buffer.")
    _LIB[gpu] = lib
    return lib


def _on_gpu() -> bool:
    import jax

    return jax.default_backend() in ("gpu", "cuda")


def emit(name: str, *values, step=None, last=None) -> None:
    """Inside a traced loop: report ``values`` (up to 6 scalars) under the line ``name``.

    ``step`` (the 1-based step count) and ``last`` (True on the loop's final step) let ``log=N``
    report only every N-th step and the last one; the choice is made on the device. Call it only when
    :func:`active` -- it raises otherwise, so a call site cannot add operations to a program built without
    progress."""
    if not _ACTIVE[0]:
        raise RuntimeError("jno.utils.beacon.emit without log=: guard the call with beacon.active().")
    import jax
    import jax.numpy as jnp

    gpu = _on_gpu()
    _load(gpu)
    dt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    vals = jax.lax.stop_gradient(jnp.stack([jnp.asarray(v, dt).reshape(()) for v in values[:6]]))
    target = f"jno_beacon_emit_{'gpu' if gpu else 'cpu'}_{'f64' if dt == jnp.float64 else 'f32'}"
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((1,), jnp.int32), has_side_effect=True, vmap_method="sequential")
    tag = np.int64(_TAGS[name])
    n = _EVERY[0]
    if n is None or n <= 1 or step is None:
        call(vals, tag=tag)
        return
    due = jnp.asarray(step) % n == 0
    if last is not None:
        due = due | jnp.asarray(last)
    jax.lax.cond(due, lambda v: call(v, tag=tag), lambda v: jnp.zeros((1,), jnp.int32), vals)


_SLOT = np.dtype([("seq", "<i8"), ("tag", "<i8"), ("v", "<f8", (6,))], align=True)


class _Reader:
    """Polls the rings and hands every new emit to ``printer(name, values)`` in emit order."""

    def __init__(self, printer=None):
        self.printer = printer or (lambda name, line: _print(name, line))  # looked up per line: tests patch it
        self.next = {}  # gpu -> next emit number expected
        self.stop = threading.Event()
        self.lock = threading.Lock()  # the thread and flush() must not hand out the same line twice
        self.thread = threading.Thread(target=self._run, name="jno-beacon", daemon=True)

    def _ring(self, gpu):
        lib = _LIB.get(gpu)
        ptr = lib.jno_beacon_ring(int(gpu)) if lib is not None else None
        if not ptr:
            return None
        n = lib.jno_beacon_slots()
        buf = (ctypes.c_byte * (n * _SLOT.itemsize)).from_address(ptr)
        return np.frombuffer(buf, dtype=_SLOT)

    def poll(self):
        with self.lock:
            self._poll()

    def _poll(self):
        for gpu in (False, True):
            ring = self._ring(gpu)
            if ring is None:
                continue
            nxt = self.next.get(gpu, 1)  # a ring that appeared after the stream opened is fresh
            R = len(ring)
            while True:
                s = ring[(nxt - 1) % R]
                seq = int(s["seq"])
                if seq < nxt:  # not written yet (or being rewritten)
                    break
                if seq > nxt:  # the writer lapped the reader: only the newest R emits are still there
                    oldest = int(ring["seq"].max()) - R + 1
                    self.printer("beacon", f"{oldest - nxt} lines lost (more than {R} between two reads)")
                    nxt = oldest
                    continue
                tag, vals = int(s["tag"]), s["v"].copy()
                if int(ring[(nxt - 1) % R]["seq"]) != nxt:  # rewritten while we copied it
                    continue
                name, fmt = _FORMATS.get(tag, (f"tag {tag}", " ".join(["{%d:.4g}" % i for i in range(6)])))
                self.printer(name, fmt.format(*vals))
                nxt += 1
            self.next[gpu] = nxt

    def _run(self):
        while not self.stop.wait(_POLL):
            self.poll()

    def start(self):
        # Emits from before the reader started are not ours: start after the newest one in each ring.
        for gpu in (False, True):
            ring = self._ring(gpu)
            if ring is not None:
                self.next[gpu] = int(ring["seq"].max()) + 1
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.poll()


def _print(name, line):
    from .logger import get_logger

    get_logger().info(f"{name} {line}")


def _reader() -> _Reader:
    """The process's one reader thread, started on first use; it outlives every solve, because a solve traced
    inside another program (a ``fem.solve`` in a ``jno.core`` loss) reports when that program runs."""
    with _READER_LOCK:
        if _READER[0] is None:
            _READER[0] = _Reader()
            _READER[0].start()
        return _READER[0]


def flush(wait: bool = True) -> None:
    """Print what the device has reported so far (``wait``: after the work already dispatched has run)."""
    r = _READER[0]
    if r is None:
        return
    if wait:
        try:
            import jax

            jax.effects_barrier()
        except Exception:  # noqa: BLE001 -- best effort; the reader thread prints it anyway
            pass
    r.poll()


def parse_log(log, where: str):
    """``log=`` as ``(on, every)``: False/None -> off, True -> every step, N >= 1 -> every N-th step and the last."""
    if log is None or log is False:
        return False, None
    if log is True:
        return True, None
    if isinstance(log, int) and not isinstance(log, bool) and log >= 1:
        return True, int(log)
    raise ValueError(f"{where}(log=...) takes True, False or a whole number of steps >= 1, not {log!r}.")


@contextlib.contextmanager
def session(log, where: str = "solve"):
    """Trace the loops run inside with ``log=`` (see :func:`parse_log`); on leaving, print what has arrived.

    Without ``log=`` this changes nothing: an enclosing solve's setting (a march's Newton steps reporting
    with the march) still applies."""
    on, n = parse_log(log, where)
    if not on:
        yield
        return
    _load(_on_gpu())  # allocate the ring before the reader looks for it
    _reader()
    prev = (_ACTIVE[0], _EVERY[0])
    _ACTIVE[0], _EVERY[0] = True, n
    try:
        yield
    finally:
        _ACTIVE[0], _EVERY[0] = prev
        flush()


def logged(fn, log, where: str = "solve"):
    """``fn`` run under :func:`session` -- for a deferred solve, whose loops are traced when it is evaluated."""
    on, _ = parse_log(log, where)
    if not on:
        return fn

    def run(*a, **k):
        with session(log, where):
            return fn(*a, **k)

    return run
