"""``fem.solve(checkpoint=...)`` on the FIXED-MESH marches: a transient (``u.t``) and a load path (``tau``).

A march returns its trajectory only when ``solve()`` returns, so a run that dies returns nothing (#141).
The moving-mesh driver could already write itself down; every other march accepted ``checkpoint=`` and
dropped it -- the directory stayed empty, which is exactly the failure checkpointing exists to prevent.

Checkpointing runs these compiled loops as a host loop of chunks of ``every`` steps, writing the frames
(one ``.npy`` memory map) and the loop carry after each chunk. Oracles:

* **same answer** -- the transient trajectory is bit-identical to the unchecked march; the load path agrees
  to round-off (its single scan and the chunked one are compiled separately, measured 2.8e-17).
* **a killed run resumes** -- a run stopped after its second write continues from that write: only the
  remaining chunks are written again, and the result is the reference.
* **fail loud** -- every path that cannot write itself down raises instead of leaving the directory empty:
  a steady solve, an adaptive time step, an evaluation under jit/grad, a parametric load path, adaptive
  load stepping, and a store that belongs to a different march.
"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver import march_checkpoint as mc

n = jno.np


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(n_t=21, reaction=0.0, alpha=None):
    """Heat on the unit square, f = 1, u0 = sin(pi x); ``reaction`` adds a cubic sink (a NONLINEAR march,
    which is judged per step), ``alpha`` makes the diffusivity a runtime parameter."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain(time=(0.0, 0.2, n_t))
    u, v = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    k = 1.0 if alpha is None else alpha
    weak = ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi
    if reaction:
        weak = weak + reaction * ui**3 * vi
    return jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - n.sin(np.pi * ci[0])])


def _load_path(n_tau=13):
    """A membrane whose stiffness hardens with the accumulated displacement ``s``: path-dependent, so a
    resume that dropped the history buffers would land somewhere else."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(tau=(0.0, 1.0, n_tau))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    return jno.fem(
        [
            (1.0 + 3.0 * s.i(-1)) * n.inner(n.grad(u, X), n.grad(phi, X), 1) - 10.0 * tau * phi,
            s.evolves(s.i(-1) + u),
            u(*cb) - 0.0,
        ]
    )


class _Killed(Exception):
    pass


def _kill_after(monkeypatch, n_writes):
    """Make the store's ``save`` raise right after its ``n_writes``-th write -- what a kill leaves on disk."""
    orig, seen = mc.FixedMarchCheckpoint.save, []

    def save(self, step, payload):
        orig(self, step, payload)
        seen.append(step)
        if len(seen) == n_writes:
            raise _Killed(step)

    monkeypatch.setattr(mc.FixedMarchCheckpoint, "save", save)


def _record_writes(monkeypatch):
    orig, seen = mc.FixedMarchCheckpoint.save, []

    def save(self, step, payload):
        orig(self, step, payload)
        seen.append(step)

    monkeypatch.setattr(mc.FixedMarchCheckpoint, "save", save)
    return seen


# ---------------------------------------------------------------------------------------------------
# transient
# ---------------------------------------------------------------------------------------------------


def test_a_checkpointed_transient_is_bit_identical_and_on_disk(tmp_path):
    ref = np.asarray(_heat().solve().fn())
    got = _heat().solve(checkpoint=jno.solve.checkpoint(str(tmp_path), every=6)).fn()
    assert isinstance(got, np.memmap), "keep='last' returns the store's memory map, not a copy in RAM"
    assert np.array_equal(np.asarray(got), ref)
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["complete"] and man["step"] == 20 and man["kind"] == "transient"
    assert np.array_equal(np.load(tmp_path / "frames.npy"), ref)

    kept = _heat().solve(checkpoint=jno.solve.checkpoint(str(tmp_path / "all"), every=6, keep="all")).fn()
    assert type(kept) is np.ndarray and np.array_equal(kept, ref)


@pytest.mark.parametrize("reaction", [0.0, 5.0], ids=["linear", "nonlinear"])
def test_a_killed_transient_resumes_from_its_last_write(tmp_path, monkeypatch, reaction):
    ref = np.asarray(_heat(reaction=reaction).solve().fn())
    spec = jno.solve.checkpoint(str(tmp_path), every=6)
    _kill_after(monkeypatch, 2)
    with pytest.raises(_Killed):
        _heat(reaction=reaction).solve(checkpoint=spec).fn()
    assert json.loads((tmp_path / "manifest.json").read_text())["step"] == 12
    monkeypatch.undo()

    writes = _record_writes(monkeypatch)
    got = np.asarray(_heat(reaction=reaction).solve(checkpoint=spec).fn())
    # It RESUMED: a fresh start would write at 6 and 12 again and still return the reference.
    assert writes == [18, 20], f"expected only the remaining chunks to be written, got writes at {writes}"
    assert np.array_equal(got, ref)


def test_bdf2_checkpoints_too(tmp_path):
    ref = np.asarray(_heat().solve(time=jno.solve.bdf2()).fn())
    got = np.asarray(_heat().solve(time=jno.solve.bdf2(), checkpoint=jno.solve.checkpoint(str(tmp_path), every=5)).fn())
    assert np.array_equal(got, ref)


def test_a_finished_run_starts_over_and_resume_false_ignores_a_partial_one(tmp_path, monkeypatch):
    spec = jno.solve.checkpoint(str(tmp_path), every=6)
    _heat().solve(checkpoint=spec).fn()
    writes = _record_writes(monkeypatch)
    _heat().solve(checkpoint=spec).fn()
    assert writes == [6, 12, 18, 20], "a COMPLETE store is not resumed"

    monkeypatch.undo()
    _kill_after(monkeypatch, 1)
    with pytest.raises(_Killed):
        _heat().solve(checkpoint=spec).fn()
    monkeypatch.undo()
    writes = _record_writes(monkeypatch)
    _heat().solve(checkpoint=jno.solve.checkpoint(str(tmp_path), every=6, resume=False)).fn()
    assert writes == [6, 12, 18, 20]


def test_a_store_from_a_different_march_is_not_spliced_onto_this_one(tmp_path, monkeypatch):
    _kill_after(monkeypatch, 1)
    with pytest.raises(_Killed):
        _heat().solve(checkpoint=jno.solve.checkpoint(str(tmp_path), every=6)).fn()
    monkeypatch.undo()
    with pytest.raises(ValueError, match="DIFFERENT march"):
        _heat(reaction=5.0).solve(checkpoint=jno.solve.checkpoint(str(tmp_path), every=6)).fn()
    with pytest.raises(ValueError, match="holds a 'transient' checkpoint"):
        _load_path().solve(checkpoint=jno.solve.checkpoint(str(tmp_path), every=4))


# ---------------------------------------------------------------------------------------------------
# load path
# ---------------------------------------------------------------------------------------------------


def test_a_checkpointed_load_path_matches_and_resumes(tmp_path, monkeypatch):
    ref = np.asarray(_load_path().solve())
    got = np.asarray(_load_path().solve(checkpoint=jno.solve.checkpoint(str(tmp_path / "a"), every=4)))
    scale = np.abs(ref).max()
    assert scale > 0.1
    assert np.abs(got - ref).max() <= 1e-14 * scale

    spec = jno.solve.checkpoint(str(tmp_path / "b"), every=4)
    _kill_after(monkeypatch, 2)
    with pytest.raises(_Killed):
        _load_path().solve(checkpoint=spec)
    monkeypatch.undo()
    writes = _record_writes(monkeypatch)
    got = np.asarray(_load_path().solve(checkpoint=spec))
    assert writes == [12, 13], f"expected only the remaining chunks, got writes at {writes}"
    assert np.abs(got - ref).max() <= 1e-14 * scale


# ---------------------------------------------------------------------------------------------------
# fail loud
# ---------------------------------------------------------------------------------------------------


def test_a_steady_solve_refuses_checkpoint(tmp_path):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xb, yb) - 0.0])
    with pytest.raises(ValueError, match="no march to write down"):
        fem.solve(checkpoint=jno.solve.checkpoint(str(tmp_path)))


def test_paths_that_cannot_write_themselves_down_raise(tmp_path):
    with pytest.raises(NotImplementedError, match="did not run through a march that can checkpoint"):
        _heat().solve(time=jno.solve.theta(1.0).adaptive(rtol=1e-3), checkpoint=jno.solve.checkpoint(str(tmp_path))).fn()
    with pytest.raises(NotImplementedError, match="did not run through a march that can checkpoint"):
        _heat().solve(lambda block, args, ts: np.zeros((len(ts), 36)), checkpoint=jno.solve.checkpoint(str(tmp_path))).fn()
    assert not (tmp_path / "frames.npy").exists()


def test_a_traced_evaluation_refuses_checkpoint(tmp_path):
    alpha = n.parameter((1,), name="alpha")
    alpha.initialize(jax.nn.initializers.constant(1.0))
    node = _heat(alpha=alpha).solve(checkpoint=jno.solve.checkpoint(str(tmp_path)))
    with pytest.raises(NotImplementedError, match="under a trace"):
        jax.grad(lambda a: node.fn(a).sum())(np.ones(1))


def test_the_load_path_legs_with_their_own_loop_refuse_checkpoint(tmp_path):
    with pytest.raises(NotImplementedError, match="adaptive load stepping"):
        _load_path().solve(tau=jno.solve.adaptive(limit=0.05), checkpoint=jno.solve.checkpoint(str(tmp_path)))
