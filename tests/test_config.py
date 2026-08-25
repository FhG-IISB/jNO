"""Tests for jno.utils.config — get_seed() and load_config()."""

import os

import pytest

import jno.utils.config as cfg_module
from jno.utils.config import (
    get_rsa_private_key,
    get_rsa_public_key,
    get_runs_base_dir,
    get_seed,
    load_config,
)

# ======================================================================
# get_seed() — reads from the cached _CONFIG dict
# ======================================================================


class TestGetSeed:
    def test_returns_default_when_config_empty(self, monkeypatch):
        """get_seed() returns 42 when no [jno] section is present."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {})
        assert get_seed() == 42

    def test_returns_seed_value(self, monkeypatch):
        """get_seed() returns the integer seed from [jno] section."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": 42}})
        assert get_seed() == 42

    def test_returns_default_when_seed_key_absent(self, monkeypatch):
        """get_seed() returns 42 when [jno] exists but has no 'seed' key."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"other_key": "x"}})
        assert get_seed() == 42

    def test_seed_is_int(self, monkeypatch):
        """get_seed() returns the value as-is from TOML (integer)."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": 7}})
        result = get_seed()
        assert isinstance(result, int)
        assert result == 7

    def test_different_seeds(self, monkeypatch):
        """get_seed() reflects whatever value is stored."""
        for seed in (0, 1, 100, 99999):
            monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": seed}})
            assert get_seed() == seed

    def test_env_seed_overrides_config(self, monkeypatch):
        monkeypatch.setenv("JNO_SEED", "123")
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": 42}})
        assert get_seed() == 123

    def test_env_seed_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("JNO_SEED", "abc")
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": 42}})
        with pytest.raises(ValueError, match="Invalid JNO_SEED"):
            get_seed()


class TestEnvOverrides:
    def test_log_dir_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("JNO_LOG_DIR", "~/my_jno_logs")
        monkeypatch.setattr(cfg_module, "_CONFIG", {"runs": {"base_dir": "./runs_from_config"}})
        assert get_runs_base_dir() == os.path.expanduser("~/my_jno_logs")

    def test_rsa_path_env_sets_both_key_paths(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JNO_RSA_PATH", str(tmp_path))
        monkeypatch.setattr(
            cfg_module,
            "_CONFIG",
            {
                "rsa": {
                    "public_key": "~/config_public.pem",
                    "private_key": "~/config_private.pem",
                }
            },
        )
        assert get_rsa_public_key() == str((tmp_path / "public.pem").resolve())
        assert get_rsa_private_key() == str((tmp_path / "private.pem").resolve())

    def test_explicit_rsa_key_env_overrides_rsa_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JNO_RSA_PATH", str(tmp_path))
        monkeypatch.setenv("JNO_RSA_PUBLIC_KEY", "~/pub_env.pem")
        monkeypatch.setenv("JNO_RSA_PRIVATE_KEY", "~/priv_env.pem")
        monkeypatch.setattr(cfg_module, "_CONFIG", {})
        assert get_rsa_public_key() == os.path.expanduser("~/pub_env.pem")
        assert get_rsa_private_key() == os.path.expanduser("~/priv_env.pem")


# ======================================================================
# load_config() — reads from the filesystem
# ======================================================================


class TestLoadConfig:
    def test_empty_when_no_toml_file(self, tmp_path, monkeypatch):
        """load_config() returns {} when no .jno.toml exists."""
        monkeypatch.chdir(tmp_path)
        result = load_config(force=True)
        assert result == {}

    def test_reads_seed_from_toml(self, tmp_path, monkeypatch):
        """load_config() parses the seed value from a .jno.toml file."""
        (tmp_path / ".jno.toml").write_bytes(b"[jno]\nseed = 99\n")
        monkeypatch.chdir(tmp_path)
        result = load_config(force=True)
        assert result.get("jno", {}).get("seed") == 99

    def test_reads_other_sections(self, tmp_path, monkeypatch):
        """load_config() parses non-jno sections correctly."""
        (tmp_path / ".jno.toml").write_bytes(b'[runs]\nbase_dir = "./my_runs"\n')
        monkeypatch.chdir(tmp_path)
        result = load_config(force=True)
        assert result.get("runs", {}).get("base_dir") == "./my_runs"

    def test_caches_result(self, tmp_path, monkeypatch):
        """A second call without force=True returns the cached value."""
        (tmp_path / ".jno.toml").write_bytes(b"[jno]\nseed = 11\n")
        monkeypatch.chdir(tmp_path)
        first = load_config(force=True)
        # Modify the file — cached result should still be returned
        (tmp_path / ".jno.toml").write_bytes(b"[jno]\nseed = 999\n")
        second = load_config(force=False)
        assert first is second

    def test_force_re_reads(self, tmp_path, monkeypatch):
        """force=True causes the file to be re-read."""
        (tmp_path / ".jno.toml").write_bytes(b"[jno]\nseed = 11\n")
        monkeypatch.chdir(tmp_path)
        load_config(force=True)
        (tmp_path / ".jno.toml").write_bytes(b"[jno]\nseed = 22\n")
        result = load_config(force=True)
        assert result["jno"]["seed"] == 22


# ======================================================================
# End-to-end: write TOML → load_config → get_seed
# ======================================================================


class TestEndToEnd:
    def test_get_seed_after_load(self, tmp_path, monkeypatch):
        """Writing a TOML then calling load_config + get_seed returns the seed."""
        (tmp_path / ".jno.toml").write_bytes(b"[jno]\nseed = 55\n")
        monkeypatch.chdir(tmp_path)
        load_config(force=True)
        assert get_seed() == 55

    def test_get_seed_returns_default_after_empty_load(self, tmp_path, monkeypatch):
        """get_seed() returns 42 when the loaded config has no seed."""
        (tmp_path / ".jno.toml").write_bytes(b"[runs]\nbase_dir = './r'\n")
        monkeypatch.chdir(tmp_path)
        load_config(force=True)
        assert get_seed() == 42


# ======================================================================
# setup(compile_cache=...) — the persistent XLA compilation cache
# ======================================================================
#
# OFF by default, deliberately: a library should not write to a user's disk uninvited, and the run
# that POPULATES the cache is slower than no cache at all, so a single cold run would be a straight
# loss. Worth opting into for anything run more than once — measured on 3-D Poisson at 27,833 nodes,
# first build 4.75 s -> 2.22 s and repeat build 2.48 s -> 1.51 s.


def _cache_dir():
    import jax

    return jax.config.jax_compilation_cache_dir


@pytest.fixture
def _restore_cache_dir():
    import jax

    prev = jax.config.jax_compilation_cache_dir
    try:
        yield
    finally:
        jax.config.update("jax_compilation_cache_dir", prev)


def test_setup_turns_the_compile_cache_on_by_default(tmp_path, monkeypatch, _restore_cache_dir):
    """The default FLIPPED (2026-08-15): the warm-cache build measured 2.1x (4.3x on Stokes) and
    jNO's normal life is the repeated run, so the persistent cache is ON unless opted out —
    `JNO_COMPILE_CACHE=0`, `[jno] compile_cache = false`, or `setup(compile_cache=False)`. This test
    used to pin the opposite ('must not write to a user's disk unless asked'); the populate-cost
    caveat it guarded is now documented in `enable_compile_cache` and docs/fem/limitations.md instead."""
    import jax

    import jno

    monkeypatch.chdir(tmp_path)  # no .jno.toml here, so nothing can opt OUT behind the test's back
    monkeypatch.delenv("JNO_COMPILE_CACHE", raising=False)
    cfg_module._CONFIG = None
    jax.config.update("jax_compilation_cache_dir", None)
    try:
        jno.setup(str(tmp_path / "script.py"), name="cache_default")
        assert _cache_dir() is not None, "the compile cache is ON by default since the 08-15 flip"
    finally:
        cfg_module._CONFIG = None


def test_setup_compile_cache_false_disables_the_default(tmp_path, monkeypatch, _restore_cache_dir):
    """The explicit opt-out must actively disable what the import-time default turned on."""
    import jax

    import jno

    monkeypatch.chdir(tmp_path)
    cfg_module._CONFIG = None
    try:
        jno.setup(str(tmp_path / "script.py"), name="cache_off", compile_cache=False)
        assert _cache_dir() is None, "setup(compile_cache=False) must disable the cache"
    finally:
        cfg_module._CONFIG = None
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def test_compile_cache_true_turns_it_on(tmp_path, _restore_cache_dir):
    import jax

    import jno

    jax.config.update("jax_compilation_cache_dir", None)
    jno.setup(str(tmp_path / "script.py"), name="cache_on", compile_cache=True)
    assert _cache_dir() is not None


def test_compile_cache_accepts_an_explicit_directory(tmp_path, _restore_cache_dir):
    import jax

    import jno

    jax.config.update("jax_compilation_cache_dir", None)
    target = tmp_path / "mycache"
    jno.setup(str(tmp_path / "script.py"), name="cache_dir", compile_cache=str(target))
    assert _cache_dir() == str(target)


def test_toml_can_turn_the_compile_cache_on(tmp_path, monkeypatch, _restore_cache_dir):
    """`[jno] compile_cache = true` is the per-project opt-in — set once, rather than editing every
    script that calls setup()."""
    import jax

    import jno

    (tmp_path / ".jno.toml").write_text("[jno]\ncompile_cache = true\n")
    monkeypatch.chdir(tmp_path)
    cfg_module._CONFIG = None  # force a re-read of the TOML from the new cwd
    jax.config.update("jax_compilation_cache_dir", None)
    try:
        jno.setup(str(tmp_path / "script.py"), name="cache_toml")
        assert _cache_dir() is not None
    finally:
        cfg_module._CONFIG = None
