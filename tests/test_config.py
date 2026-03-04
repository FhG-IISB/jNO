"""Tests for jno.utils.config — get_seed() and load_config()."""

import pytest

import jno.utils.config as cfg_module
from jno.utils.config import get_seed, load_config


# ======================================================================
# get_seed() — reads from the cached _CONFIG dict
# ======================================================================


class TestGetSeed:
    def test_returns_none_when_config_empty(self, monkeypatch):
        """get_seed() returns None when no [jno] section is present."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {})
        assert get_seed() is None

    def test_returns_seed_value(self, monkeypatch):
        """get_seed() returns the integer seed from [jno] section."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": 42}})
        assert get_seed() == 42

    def test_returns_none_when_seed_key_absent(self, monkeypatch):
        """get_seed() returns None when [jno] exists but has no 'seed' key."""
        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"other_key": "x"}})
        assert get_seed() is None

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

    def test_get_seed_returns_none_after_empty_load(self, tmp_path, monkeypatch):
        """get_seed() returns None when the loaded config has no seed."""
        (tmp_path / ".jno.toml").write_bytes(b"[runs]\nbase_dir = './r'\n")
        monkeypatch.chdir(tmp_path)
        load_config(force=True)
        assert get_seed() is None
