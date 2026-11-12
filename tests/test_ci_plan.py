"""The CI shard planner (.github/ci_plan.py) runs every test file exactly once."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / ".github" / "ci_plan.py"
_spec = importlib.util.spec_from_file_location("ci_plan", _PATH)
ci_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_plan)


def _planned(capsys, *groups):
    ci_plan.main(["matrix", *groups])
    return json.loads(capsys.readouterr().out)["include"]


def test_every_file_lands_in_exactly_one_shard(capsys):
    seen = []
    for job in _planned(capsys):
        seen += ci_plan.shard_files(job["group"], job["shard"], job["of"])
    assert sorted(seen) == ci_plan.test_files()


def test_core_is_the_complement_of_the_named_groups():
    assert ci_plan.group_of("tests/test_fdm.py") == "fdm"  # the headline file, no underscore
    assert ci_plan.group_of("tests/test_fem_eigs.py") == "fem"
    assert ci_plan.group_of("tests/test_a_file_nobody_has_written.py") == "core"


def test_an_unmeasured_file_is_still_planned(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_plan, "TIMES", tmp_path / "missing.tsv")
    files = [f for _, f in ci_plan.timed_files("rcwa")]
    assert files and ci_plan.shard_files("rcwa", 1, 1) == sorted(files)


def test_shards_are_balanced():
    """Longest-first dealing evens the shards out to within a few percent; dealing in name order
    instead leaves the slowest shard well above the mean, and that shard sets the PR's wait."""
    times = ci_plan.read_times()
    for group in ("fem", "core"):
        n = ci_plan.n_shards(group)
        assert n > 1
        loads = [sum(times.get(f, 0) for f in ci_plan.shard_files(group, k, n)) for k in range(1, n + 1)]
        assert max(loads) <= 1.05 * sum(loads) / n
        assert max(loads) <= 60 * ci_plan.TARGET_MINUTES


def test_an_unknown_group_is_refused():
    with pytest.raises(SystemExit):
        ci_plan.main(["matrix", "nope"])
