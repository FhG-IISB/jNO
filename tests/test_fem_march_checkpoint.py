"""A checkpointed march writes its frames as it goes, and can be restarted from them.

``fem.solve(adapt=...)`` holds every frame in memory and hands them back only when it returns, so a
run that dies -- OOM, a kill, a power cut -- yields NOTHING however far it got. ``checkpoint=`` writes
frames to disk as they are produced, drops them from memory, and records what the march needs to
continue.

The restart is not new machinery: a topology rebuild already re-enters the march with
``{"start", "old": (points, cells, state, layout), "budget", "carry"}``. That tuple is the
checkpoint; this only writes it down -- plus the domain the next segment starts ON, which is NOT the
same mesh as ``old`` once a rebuild has retriangulated it.

Oracles here:
  * a checkpointed march returns the SAME trajectory, frame for frame, as one without -- exactly,
    not approximately, since checkpointing must not perturb the arithmetic;
  * the frames really are on disk (chunk files exist) and the returned columns load lazily;
  * a march interrupted after a rebuild resumes and lands on the reference trajectory;
  * ``checkpoint=`` without ``adapt=`` is REFUSED rather than silently doing nothing.

The two drops merge during the march, so the rebuild path -- the one that flushes and hands the
writer down the resume chain -- is exercised, not just the periodic flush.
"""

import json
import os

import jax
import numpy as np
import pytest

import jno

H = 0.08


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _drops(*, adapt=None, checkpoint=None, T=0.32, n=13):
    """Two disks closing a gap; the same problem as `test_fem_droplets_merge`, marched to a merge."""
    d = (jno.shape.disk(0.0, 0.0, 0.5, size=H) | jno.shape.disk(1.35, 0.0, 0.5, size=H)).domain(time=(0.0, T, n))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),
            u(ci[0], ci[1]) - jno.np.tanh(8.0 * (ci[0] - 0.675)),
            xi.d(ti) + 0.5 * jno.np.tanh(8.0 * (xi - 0.675)),
        ]
    )
    kw = {} if checkpoint is None else {"checkpoint": checkpoint}
    return fem.solve(adapt=adapt, **kw)


def _same(a, b):
    """Frame-for-frame equality of two trajectories -- states AND meshes, exactly."""
    assert len(a) == len(b), f"{len(a)} frames vs {len(b)}"
    assert np.allclose(np.asarray(a.times), np.asarray(b.times)), "time grids differ"
    for i in range(len(a)):
        sa, sb = np.asarray(a.states[i]), np.asarray(b.states[i])
        assert sa.shape == sb.shape, f"frame {i}: state shape {sa.shape} vs {sb.shape}"
        assert np.array_equal(sa, sb), f"frame {i}: max |d| = {np.abs(sa - sb).max():.3e}"
        (pa, ca), (pb, cb) = a.meshes[i], b.meshes[i]
        assert np.array_equal(np.asarray(pa), np.asarray(pb)), f"frame {i}: points differ"
        assert np.array_equal(np.asarray(ca), np.asarray(cb)), f"frame {i}: cells differ"


def test_a_checkpointed_march_returns_exactly_the_same_trajectory(tmp_path):
    """Checkpointing is bookkeeping: it must not change a single number the march produces."""
    adapt = jno.solve.remesh(alpha=1.2, every=1)
    ref = _drops(adapt=adapt)
    got = _drops(adapt=adapt, checkpoint=jno.solve.checkpoint(str(tmp_path / "ck"), every=3))
    _same(ref, got)


def test_the_frames_are_written_to_disk_and_read_back_lazily(tmp_path):
    """The point of ``keep='last'``: frames live in chunk files, not in the returned lists."""
    store = tmp_path / "ck"
    traj = _drops(adapt=jno.solve.remesh(alpha=1.2, every=1), checkpoint=jno.solve.checkpoint(str(store), every=3))

    chunks = sorted(f for f in os.listdir(store) if f.startswith("frames_"))
    assert chunks, "no chunk files were written"
    man = json.loads((store / "manifest.json").read_text())
    assert man["complete"] is True, "a finished march must mark its manifest complete"
    assert man["n_frames"] == len(traj), f"manifest says {man['n_frames']} frames, trajectory has {len(traj)}"

    # lazy, not a plain list -- and indexable both ways round
    assert not isinstance(traj.states, list)
    assert np.array_equal(np.asarray(traj.states[-1]), np.asarray(traj.states[len(traj) - 1]))
    assert len(traj.meshes) == len(traj) and len(traj.layouts) == len(traj)

    # keep="all" materialises instead, for callers that would rather hold it in RAM
    eager = _drops(
        adapt=jno.solve.remesh(alpha=1.2, every=1),
        checkpoint=jno.solve.checkpoint(str(tmp_path / "ck_all"), every=3, keep="all"),
    )
    assert isinstance(eager.states, list)
    _same(traj, eager)


def test_an_interrupted_march_resumes_onto_the_reference_trajectory(tmp_path):
    """Cut the store back to its last rebuild -- what a crash leaves -- and march again.

    ``latest.npz`` is rewritten at every rebuild, so after truncating the frames written *after* the
    final rebuild the store is exactly what an interrupted run would have on disk. Re-solving with the
    same ``checkpoint=`` must continue from there and land on the reference, not restart from zero.
    """
    adapt = jno.solve.remesh(alpha=1.2, every=1)
    ref = _drops(adapt=adapt)

    store = tmp_path / "ck"
    _drops(adapt=adapt, checkpoint=jno.solve.checkpoint(str(store), every=3))
    man = json.loads((store / "manifest.json").read_text())
    if not (store / "latest.npz").exists():
        pytest.skip("this march never rebuilt, so there is no mid-flight checkpoint to resume from")

    # roll back to the last rebuild boundary: drop the trailing chunk and un-complete the manifest
    assert len(man["chunks"]) >= 2, f"need >1 chunk to truncate, got {len(man['chunks'])}"
    dropped = man["chunks"].pop()
    os.remove(store / dropped["file"])
    man["n_frames"] = man["chunks"][-1]["stop"]
    man["complete"] = False
    (store / "manifest.json").write_text(json.dumps(man))

    kept = store / man["chunks"][0]["file"]
    before = (kept.stat().st_mtime_ns, kept.read_bytes())

    resumed = _drops(adapt=adapt, checkpoint=jno.solve.checkpoint(str(store), every=3))
    _same(ref, resumed)

    # ...and it RESUMED rather than starting over, which would produce the same trajectory and pass
    # vacuously. A fresh start resets the chunk index and rewrites frames_000000.npz; a resume leaves
    # every surviving chunk untouched and appends after it.
    assert kept.exists(), "resume destroyed an existing chunk"
    assert (kept.stat().st_mtime_ns, kept.read_bytes()) == before, "frames_000000 was rewritten: it restarted"
    man2 = json.loads((store / "manifest.json").read_text())
    assert man2["chunks"][0] == man["chunks"][0], "the surviving chunk index was not preserved"
    assert man2["n_frames"] == len(ref), f"resumed store has {man2['n_frames']} frames, reference has {len(ref)}"


def test_a_finished_run_is_not_resumed(tmp_path):
    """A complete manifest means there is nothing to continue; re-solving starts over cleanly."""
    adapt = jno.solve.remesh(alpha=1.2, every=1)
    store = tmp_path / "ck"
    first = _drops(adapt=adapt, checkpoint=jno.solve.checkpoint(str(store), every=3))
    again = _drops(adapt=adapt, checkpoint=jno.solve.checkpoint(str(store), every=3))
    _same(first, again)


def test_checkpoint_without_adapt_is_refused(tmp_path):
    """One compiled scan has no host-side boundary to write at -- say so instead of writing nothing."""
    with pytest.raises(NotImplementedError, match="adapt="):
        _drops(adapt=None, checkpoint=jno.solve.checkpoint(str(tmp_path / "ck")))


def test_the_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="keep"):
        jno.solve.checkpoint("/tmp/x", keep="sometimes")
    with pytest.raises(ValueError, match="every"):
        jno.solve.checkpoint("/tmp/x", every=0)
