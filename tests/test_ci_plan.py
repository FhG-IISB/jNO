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


# ------------------------------------------------------------------------------------------------
# The compile cache of the per-file runner (.github/ci_test_per_file.sh)
# ------------------------------------------------------------------------------------------------

_GITHUB = _PATH.parent


def test_the_runner_does_not_cap_the_compile_cache_through_jax():
    """``JAX_COMPILATION_CACHE_MAX_SIZE`` makes JAX rescan the whole cache directory on every write, so
    filling a cold cache costs time quadratic in the number of entries (one test file writes 1,560), and
    every nightly shard that started cold ran into its timeout."""
    import re

    text = (_GITHUB / "ci_test_per_file.sh").read_text()
    assert not re.search(r"^\s*export\s+JAX_COMPILATION_CACHE_MAX_SIZE", text, re.M)
    assert "ci_prune_cache.sh" in text, "the cache has no size bound left"


def _cache(tmp_path, n=5, kib=300):
    import os

    for k in range(n):
        f = tmp_path / f"entry{k}-cache"
        f.write_bytes(b"x" * kib * 1024)
        os.utime(f, (1_000_000 + k, 1_000_000 + k))  # entry0 is the oldest
    return tmp_path


def _prune(directory, max_mb):
    import subprocess

    subprocess.run(["bash", str(_GITHUB / "ci_prune_cache.sh"), str(directory), str(max_mb)], check=True)
    return sorted(p.name for p in directory.iterdir())


def test_pruning_removes_the_oldest_entries_until_the_cache_fits(tmp_path):
    assert _prune(_cache(tmp_path), 1) == ["entry2-cache", "entry3-cache", "entry4-cache"]  # 900 KiB of 1 MiB


def test_pruning_leaves_a_cache_under_the_limit_alone(tmp_path):
    assert len(_prune(_cache(tmp_path), 2)) == 5


def test_pruning_a_directory_that_does_not_exist_is_not_an_error(tmp_path):
    import subprocess

    subprocess.run(["bash", str(_GITHUB / "ci_prune_cache.sh"), str(tmp_path / "missing"), "1"], check=True)


# ------------------------------------------------------------------------------------------------
# A shard that runs out of time (.github/workflows/ci.yml, nightly.yml)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("workflow", "job"), [("ci.yml", "test"), ("nightly.yml", "full")])
def test_a_shard_that_runs_out_of_time_still_saves_its_cache_and_file_times(workflow, job):
    """A job that reaches ITS timeout is cancelled, and the cache save and the file-time upload (both
    ``if: '!cancelled()'``) are skipped: the next run starts cold again and nothing says which files were
    slow. The test step therefore has its own, shorter limit. A step that times out FAILS, and the steps
    after it still run."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((_GITHUB / "workflows" / workflow).read_text())["jobs"][job]
    steps = spec["steps"]
    (test,) = [k for k, s in enumerate(steps) if "ci_test_per_file.sh" in s.get("run", "")]
    assert steps[test].get("timeout-minutes", float("inf")) <= spec["timeout-minutes"] - 10
    after = [s.get("uses", "").split("@")[0] for s in steps[test + 1 :]]
    assert "actions/cache/save" in after and "actions/upload-artifact" in after
    for s in steps[test + 1 :]:
        assert s.get("if") == "!cancelled()", f"{s.get('uses')} is skipped when the test step fails"
