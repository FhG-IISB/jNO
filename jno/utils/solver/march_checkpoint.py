"""Checkpointing for a moving-mesh march: write frames to disk as they are produced, drop them
from memory, and be able to restart a killed run from the last write.

WHY this exists. A ``fem.solve(adapt=...)`` march holds every frame in memory and returns them all
at once, so a run that dies -- OOM, a kill, a power cut -- returns NOTHING, however far it got.
Measured on a laser-melt study: three separate runs were lost this way, one of them at 91 % (19968
of 22000 steps, 4.35 ms of 4.80 ms) after half an hour of correct physics.

WHAT makes it cheap. The march already knows how to restart itself mid-flight: every topology
rebuild re-enters :func:`run_mesh_motion` with

    _resume = {"start": i, "old": (points, cells, state, layout), "budget": ..., "carry": ...}

That tuple IS a checkpoint -- it is simply never written down. This module persists it, plus the
frames produced so far, and hands back a trajectory whose frames load from disk on demand.

LAYOUT on disk::

    <path>/
      manifest.json        dt, t0/t1, n_steps, dim, chunk index, `complete` flag
      latest.npz           points, cells, state, start, carry, budget  <- the resume payload
      frames_000000.npz    t, s0..sk, p0..pk, c0..ck, layout  (one chunk)
      frames_000500.npz    ...

A chunk never spans a rebuild, so every frame in it shares one field layout and one connectivity;
that is what lets the layout be stored once per chunk instead of once per frame.

The per-frame arrays are stored separately (``s0``, ``s1``, ...) rather than stacked, because a
rebuild CHANGES the node count -- ``n_dofs`` is not constant along a march, so there is no single
rectangular array to stack into.
"""

from __future__ import annotations

import contextvars as _contextvars
import hashlib as _hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _obj(x):
    """A 0-d object array holding ``x``.

    ``np.array(x, dtype=object)`` is NOT this when ``x`` is a sequence: a 4-tuple becomes a shape-(4,)
    array whose ``.item()`` then raises. The remesh budget IS a tuple, so writing it the obvious way
    made every resume fail to read back -- and the read was wrapped in a broad ``except``, so the march
    silently started over instead. Wrap explicitly.
    """
    a = np.empty((), dtype=object)
    a[()] = x
    return a


@dataclass(frozen=True)
class CheckpointSpec:
    """What :func:`jno.solve.checkpoint` builds; see there for the user-facing documentation."""

    path: str
    every: int = 500
    keep: str = "last"  # "last" -> flush and DROP frames; "all" -> flush but keep them in memory
    resume: bool = True

    def __post_init__(self):
        if self.keep not in ("last", "all"):
            raise ValueError(f"checkpoint(keep=): expected 'last' or 'all', got {self.keep!r}.")
        if int(self.every) < 1:
            raise ValueError(f"checkpoint(every=): must be >= 1, got {self.every}.")


class _Frames(Sequence):
    """One lazily-loaded column of the trajectory (``states`` / ``meshes`` / ``layouts``).

    Indexing loads the containing chunk and caches it, so sequential iteration -- which is what
    ``resample`` and every plotting loop does -- touches each file once. Only the chunks in flight
    are resident, which is the whole point: the trajectory no longer has to fit in RAM.
    """

    __slots__ = ("_dir", "_index", "_kind", "_cache", "_cache_n")

    def __init__(self, directory: str, index: list[dict], kind: str, cache_n: int = 2):
        self._dir, self._index, self._kind = directory, index, kind
        self._cache: dict[str, Any] = {}
        self._cache_n = cache_n

    def __len__(self) -> int:
        return self._index[-1]["stop"] if self._index else 0

    def _load(self, fname: str):
        z = self._cache.get(fname)
        if z is None:
            z = dict(np.load(os.path.join(self._dir, fname), allow_pickle=True))
            if len(self._cache) >= self._cache_n:  # FIFO: sequential reads only ever need the tail
                self._cache.pop(next(iter(self._cache)))
            self._cache[fname] = z
        return z

    def _chunk_of(self, i: int) -> tuple[dict, int]:
        for ch in self._index:
            if ch["start"] <= i < ch["stop"]:
                return ch, i - ch["start"]
        raise IndexError(f"frame {i} is not in any checkpoint chunk (have {len(self)} frames).")

    def __getitem__(self, i):
        n = len(self)
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(n))]
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"frame index {i} out of range for {n} frames.")
        ch, k = self._chunk_of(i)
        z = self._load(ch["file"])
        if self._kind == "states":
            return z[f"s{k}"]
        if self._kind == "meshes":
            return (z[f"p{k}"], z[f"c{k}"])
        return z["layout"].item()  # one layout per chunk, by construction

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class MarchCheckpoint:
    """Writer + reader for one march's checkpoint directory.

    The march appends frames as it produces them and calls :meth:`flush` at a safe boundary -- every
    ``every`` steps, and always immediately before a rebuild, since a rebuild is exactly where the
    layout changes and where a restart would have to begin anyway.
    """

    def __init__(self, spec: CheckpointSpec, *, meta: dict, reopen: bool = False):
        self.spec = spec
        self.dir = os.path.abspath(os.path.expanduser(spec.path))
        os.makedirs(self.dir, exist_ok=True)
        self._index: list[dict] = []
        self._buf: list[tuple[float, Any, Any, Any]] = []
        self.n_frames = 0  # frames written so far
        if reopen:  # continue an interrupted run: keep its chunks and append after them
            with open(os.path.join(self.dir, "manifest.json")) as fh:
                _old = json.load(fh)
            self._index = list(_old.get("chunks", []))
            self.n_frames = int(_old.get("n_frames", 0))
        self._meta = dict(meta)
        self._layout = None
        self._write_manifest(complete=False)

    # -- writing ---------------------------------------------------------------------------------

    def append(self, t: float, state, points, cells, layout) -> None:
        self._buf.append((float(t), np.asarray(state), np.asarray(points), np.asarray(cells)))
        self._layout = layout

    def pending(self) -> int:
        return len(self._buf)

    def flush(self, *, resume: dict | None = None, domain=None, complete: bool = False) -> None:
        """Write the buffered frames as one chunk, then the resume payload and the manifest.

        Order matters: the chunk lands BEFORE the manifest that references it, so a crash midway
        leaves a manifest that is merely behind, never one that points at a file which is not there.
        """
        if self._buf:
            fname = f"frames_{self.n_frames:06d}.npz"
            out: dict[str, Any] = {"t": np.array([b[0] for b in self._buf], dtype=float)}
            for k, (_, s, p, c) in enumerate(self._buf):
                out[f"s{k}"], out[f"p{k}"], out[f"c{k}"] = s, p, c
            out["layout"] = _obj(self._layout)
            np.savez(os.path.join(self.dir, fname), **out)
            self._index.append({"file": fname, "start": self.n_frames, "stop": self.n_frames + len(self._buf)})
            self.n_frames += len(self._buf)
            self._buf.clear()
        if resume is not None:
            pts, cells, state, _lay = resume["old"]
            _dom = domain if domain is not None else (pts, cells)
            np.savez(
                os.path.join(self.dir, "latest.npz"),
                points=np.asarray(pts),
                cells=np.asarray(cells),
                state=np.asarray(state),
                start=np.asarray(int(resume["start"])),
                carry=np.array(str(resume.get("carry", "identity"))),
                budget=_obj(resume.get("budget")),
                layout=_obj(_lay),
                dom_points=np.asarray(_dom[0]),
                dom_cells=np.asarray(_dom[1]),
            )
        self._write_manifest(complete=complete)

    def _write_manifest(self, *, complete: bool) -> None:
        man = dict(self._meta)
        man.update({"chunks": self._index, "n_frames": self.n_frames, "complete": bool(complete)})
        tmp = os.path.join(self.dir, "manifest.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(man, fh, indent=1)
        os.replace(tmp, os.path.join(self.dir, "manifest.json"))  # atomic: never a half-written manifest

    # -- reading ---------------------------------------------------------------------------------

    def frames(self) -> tuple[np.ndarray, _Frames, _Frames, _Frames]:
        """``(times, states, meshes, layouts)`` -- the columns an ``AdaptiveTrajectory`` wants."""
        times = []
        for ch in self._index:
            z = np.load(os.path.join(self.dir, ch["file"]), allow_pickle=True)
            times.append(np.asarray(z["t"], dtype=float))
        t = np.concatenate(times) if times else np.zeros(0)
        return (
            t,
            _Frames(self.dir, self._index, "states"),
            _Frames(self.dir, self._index, "meshes"),
            _Frames(self.dir, self._index, "layouts"),
        )


def load(path: str):
    """Open a checkpoint directory written by a previous march.

    Returns ``(manifest, resume_or_None, (times, states, meshes, layouts))``. ``resume`` is the
    payload a march needs to continue: ``{"start", "old": (points, cells, state, layout), "budget",
    "carry"}``, or ``None`` when the run finished (nothing to continue).
    """
    d = os.path.abspath(os.path.expanduser(path))
    with open(os.path.join(d, "manifest.json")) as fh:
        man = json.load(fh)
    idx = man.get("chunks", [])
    times = []
    for ch in idx:
        z = np.load(os.path.join(d, ch["file"]), allow_pickle=True)
        times.append(np.asarray(z["t"], dtype=float))
    cols = (
        np.concatenate(times) if times else np.zeros(0),
        _Frames(d, idx, "states"),
        _Frames(d, idx, "meshes"),
        _Frames(d, idx, "layouts"),
    )
    resume = None
    latest = os.path.join(d, "latest.npz")
    if not man.get("complete", False) and os.path.exists(latest):
        z = np.load(latest, allow_pickle=True)
        resume = {
            "start": int(z["start"]),
            "old": (z["points"], z["cells"], z["state"], z["layout"].item()),
            "budget": z["budget"].item(),
            "carry": str(z["carry"]),
            "domain": (z["dom_points"], z["dom_cells"]) if "dom_points" in z.files else None,
        }
    return man, resume, cols


# ------------------------------------------------------------------------------------------------
# Fixed-mesh marches: a plain transient (`u.t`) and a pseudo-time load path (`domain(tau=...)`).
# ------------------------------------------------------------------------------------------------
#
# Their frames all share one shape, so they are stored as ONE `.npy` memory map rather than per-chunk
# files: the march writes rows into it, the OS pages them out, and with ``keep="last"`` the returned
# trajectory IS that map -- frames read from disk on demand. ``latest.npz`` holds what the march needs
# to continue: the compiled loop's carry at the last written step, plus whatever the march judges its
# steps by (residual history), written atomically after the frames it refers to.
#
# These marches are compiled loops. Checkpointing runs them as a HOST loop over chunks of ``every``
# steps instead -- bit-identical arithmetic (a scan of k steps then k more IS a scan of 2k), but a chain
# of scans is not one scan, so reverse-mode differentiation through a checkpointed march is refused
# (:func:`refuse_traced`) rather than handed a gradient for a different program.


def digest(*arrays) -> str:
    """A short content hash of ``arrays`` -- how a resume recognises the march that wrote a store."""
    h = _hashlib.sha1()
    for a in arrays:
        try:
            a = np.ascontiguousarray(np.asarray(a))
        except Exception:  # noqa: BLE001 - not array-like (a pytree's static leaf): hash what it says it is
            a = np.asarray(repr(a))
        if a.dtype == object:
            h.update(repr(a.tolist()).encode())
            continue
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


class _Request:
    __slots__ = ("spec", "claimed_by")

    def __init__(self, spec: CheckpointSpec):
        self.spec, self.claimed_by = spec, None


_ACTIVE: _contextvars.ContextVar = _contextvars.ContextVar("jno_march_checkpoint", default=None)


class requested:
    """``with requested(spec): ...`` -- a ``fem.solve(checkpoint=...)`` waiting for its march to claim it.

    A march that can write itself down calls :func:`claim`; on exit, a request NOBODY claimed raises, so a
    path that cannot checkpoint (a traced evaluation, a sharded or user-supplied integrator, a scheme with
    its own loop) can never return a trajectory while the directory stays empty."""

    def __init__(self, spec: CheckpointSpec | None, what: str):
        self.req = None if spec is None else _Request(spec)
        self.what = what

    def __enter__(self):
        self._tok = _ACTIVE.set(self.req)
        return self.req

    def __exit__(self, et, ev, tb):
        _ACTIVE.reset(self._tok)
        if et is None and self.req is not None and self.req.claimed_by is None:
            raise NotImplementedError(
                f"fem.solve(checkpoint=): {self.what} did not run through a march that can checkpoint, so "
                "nothing would have been written. Checkpointing is wired for the built-in schemes evaluated "
                "eagerly (`fem.solve(...).fn()` or a fixed `time=` scheme: theta / bdf2 / sdirk / rosenbrock / "
                "exponential) and for the fixed-grid `tau` load path. It is not wired for a sharded march, a "
                "`solve_fn=` integrator of your own, an adaptive time step, or an evaluation under "
                "jit/grad/jno.core. Drop checkpoint=, or run the march eagerly with a built-in scheme."
            )
        return False


def claim(who: str) -> CheckpointSpec | None:
    """The active request's spec, claimed for ``who``; ``None`` when no checkpoint was asked for.

    A second claim in one solve raises: it means the solve runs as SEVERAL marches (a preconditioner
    refreshed between chunks, a pilot and a replay), and one store cannot hold more than one of them."""
    req = _ACTIVE.get()
    if req is None:
        return None
    if req.claimed_by is not None:
        raise NotImplementedError(
            f"fem.solve(checkpoint=): this solve runs more than one march ({req.claimed_by!r}, then {who!r}) "
            "-- e.g. a preconditioner refreshed between chunks -- and one checkpoint store holds one march. "
            "Drop checkpoint= for this configuration."
        )
    req.claimed_by = who
    return req.spec


def refuse_traced(spec: CheckpointSpec | None, values) -> None:
    """Raise when a checkpointed march would run under a trace (``jit`` / ``grad`` / ``jno.core``).

    A checkpointed march is a host loop of compiled chunks: it cannot run on tracers, and a chain of scans
    is not the single scan whose adjoint jNO differentiates. Say so instead of writing nothing."""
    if spec is None:
        return
    import jax
    from jax._src import core as _core

    traced = not _core.trace_state_clean() or any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(values))
    if traced:
        raise NotImplementedError(
            "fem.solve(checkpoint=): the march is being evaluated under a trace (jit / grad / jno.core), "
            "and a checkpointed march is a HOST loop over compiled chunks -- it can neither run on tracers "
            "nor be reverse-mode differentiated (a chain of scans is not the one scan whose adjoint jNO "
            "builds). Checkpoint the forward run instead: `fem.solve(<param>=value, checkpoint=...)`, or "
            "drop checkpoint= to differentiate."
        )


class FixedMarchCheckpoint:
    """Writer + reader for a fixed-mesh march's checkpoint directory.

    LAYOUT on disk::

        <path>/
          manifest.json   kind, meta (what the march IS), step (last written), complete
          frames.npy      (n_frames, n_dofs) -- the trajectory, a memory map filled as the march runs
          latest.npz      the loop carry at ``step`` and the march's residual history

    ``meta`` identifies the march (grid, sizes, carry structure). Resuming into a store whose meta differs
    raises rather than splicing two different problems' frames together.
    """

    def __init__(self, spec: CheckpointSpec, *, kind: str, meta: dict, shape: tuple, dtype):
        self.spec, self.kind = spec, kind
        self.dir = os.path.abspath(os.path.expanduser(spec.path))
        os.makedirs(self.dir, exist_ok=True)
        self.meta = json.loads(json.dumps(meta))  # what a manifest round-trip will hand back
        self.resumed: dict | None = None
        man_path = os.path.join(self.dir, "manifest.json")
        frames_path = os.path.join(self.dir, "frames.npy")
        old = None
        if os.path.exists(man_path):
            with open(man_path) as fh:
                old = json.load(fh)
            if old.get("kind") != kind:
                raise ValueError(
                    f"fem.solve(checkpoint=): {self.dir} holds a {old.get('kind', 'moving-mesh')!r} checkpoint, "
                    f"and this is a {kind!r} march. Point checkpoint= at a fresh directory."
                )
        if old is not None and spec.resume and not old.get("complete", False) and int(old.get("step", 0)) > 0:
            if old.get("meta") != self.meta:
                diff = sorted(
                    k for k in set(self.meta) | set(old.get("meta", {})) if self.meta.get(k) != old["meta"].get(k)
                )
                raise ValueError(
                    f"fem.solve(checkpoint=): {self.dir} holds an unfinished run of a DIFFERENT march (differs in "
                    f"{diff}), and resuming would splice its frames onto this one. Delete the directory, point "
                    "checkpoint= elsewhere, or pass resume=False to start over."
                )
            with np.load(os.path.join(self.dir, "latest.npz"), allow_pickle=False) as z:
                self.resumed = {k: z[k] for k in z.files}
            self.resumed["step"] = np.asarray(int(old["step"]))  # the step the carry is AT (from the manifest)
            self.frames = np.lib.format.open_memmap(frames_path, mode="r+")
        else:
            self.frames = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
            self._write_manifest(step=0, complete=False)

    def save(self, step: int, payload: dict) -> None:
        """Persist the frames written so far, then the carry at ``step``, then the manifest naming it.

        The order is the crash contract: a manifest never names a step whose carry or frames are not on
        disk, so a kill at any instant leaves a store that resumes from an earlier, consistent step."""
        self.frames.flush()
        tmp = os.path.join(self.dir, "latest.tmp.npz")
        np.savez(tmp, **{k: np.asarray(v) for k, v in payload.items()})
        os.replace(tmp, os.path.join(self.dir, "latest.npz"))
        self._write_manifest(step=int(step), complete=False)

    def finish(self, step: int):
        """Mark the run complete and return the trajectory: the memory map (``keep="last"``), or a copy in RAM."""
        self.frames.flush()
        self._write_manifest(step=int(step), complete=True)
        return self.frames if self.spec.keep == "last" else np.array(self.frames)

    def _write_manifest(self, *, step: int, complete: bool) -> None:
        man = {"kind": self.kind, "meta": self.meta, "step": int(step), "complete": bool(complete)}
        tmp = os.path.join(self.dir, "manifest.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(man, fh, indent=1)
        os.replace(tmp, os.path.join(self.dir, "manifest.json"))


def carry_meta(carry) -> dict:
    """The structure of a loop carry -- leaf shapes and dtypes, in order -- for a store's ``meta``.

    Not the tree's own description: its dict keys include ids minted afresh each time a form is built
    (a state field's key), so the same march rebuilt in a new process would never match its own store."""
    import jax

    leaves = jax.tree_util.tree_leaves(carry)
    return {"leaves": [[list(np.shape(x)), str(np.asarray(x).dtype)] for x in leaves]}


def pack_carry(carry, prefix: str = "carry") -> dict:
    """The carry's leaves as ``{prefix_i: host array}``."""
    import jax

    return {f"{prefix}_{i}": np.asarray(x) for i, x in enumerate(jax.tree_util.tree_leaves(carry))}


def unpack_carry(template, stored: dict, prefix: str = "carry"):
    """Rebuild a carry shaped like ``template`` from :func:`pack_carry`'s arrays (device arrays, template dtypes)."""
    import jax
    import jax.numpy as jnp

    leaves, tree = jax.tree_util.tree_flatten(template)
    new = [jnp.asarray(stored[f"{prefix}_{i}"], dtype=jnp.asarray(x).dtype) for i, x in enumerate(leaves)]
    return jax.tree_util.tree_unflatten(tree, new)
