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

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
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
