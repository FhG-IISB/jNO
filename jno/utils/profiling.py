"""Shared JAX performance profiling for jno ``.solve(profile=True)`` — used by ``jno.fem``, ``jno.fdm``
and ``jno.rcwa``, mirroring ``jno.core.solve(profile=True)``.

``profile_solve`` runs a solve **eagerly** inside a JAX Perfetto trace, prints a one-line size/time
summary, and writes the trace to ``./jno_traces`` (open at ``chrome://tracing`` or the Perfetto UI).
``annotate`` labels a solve stage in that trace (a no-op unless a profile session is active)."""

import os
import time
from contextlib import nullcontext

import jax


def annotate(name):
    """A ``jax.profiler.TraceAnnotation`` labelling a solve stage when a profile session is active; a no-op
    context otherwise, so wrapping a hot stage in it never costs anything outside profiling."""
    from jax._src import profiler as _jp

    return jax.profiler.TraceAnnotation(name) if _jp._profile_state.profile_session is not None else nullcontext()


def profile_solve(run, *, label, warm=True):
    """Run ``run()`` eagerly inside a JAX Perfetto trace, print ``label`` + wall time, write the trace to
    ``./jno_traces``, and return the result.

    ``warm`` runs ``run()`` once first so the timing excludes XLA compilation; pass ``warm=False`` when a
    re-run would be unsafe (e.g. an in-place adaptive remesh). A deferred (trace-node) result has nothing to
    block on, so the timing then reflects graph construction, not the numeric solve — profile a concrete
    forward solve for a meaningful number."""
    trace_dir = os.path.join(os.getcwd(), "jno_traces")
    os.makedirs(trace_dir, exist_ok=True)

    def _block(result):
        try:
            jax.block_until_ready(result)  # force the async DAG to finish so the timer/trace see real work
        except (TypeError, ValueError, AttributeError):
            pass  # a non-array result (a deferred trace node) has no arrays to block on
        return result

    if warm:
        _block(run())
    t0 = time.perf_counter()
    with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
        result = _block(run())
    dt = time.perf_counter() - t0
    print(f"[{label}]  wall {dt * 1e3:.1f} ms  ·  Perfetto trace → {trace_dir}")
    return result
