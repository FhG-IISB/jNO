"""Global jNO configuration and project setup.

jNO looks for a TOML config file in two locations (first match wins):

1. ``.jno.toml`` in the current working directory   (project-level)
2. ``~/.jno/config.toml``                           (user-level)

Example ``.jno.toml``::

    [jno]
    seed = 42           # global RNG seed (reproducibility)

    [runs]
    base_dir = "./runs"

    [rsa]
    public_key  = "~/.jno/public.pem"
    private_key = "~/.jno/private.pem"

All fields are optional; omitting ``[rsa]`` means save/load fall back to
unencrypted pickle, just as before.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH: Path | None = None


def _candidate_paths() -> list[Path]:
    return [
        Path.cwd() / ".jno.toml",
        Path.home() / ".jno" / "config.toml",
    ]


def _read_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(force: bool = False) -> dict[str, Any]:
    """Read and cache the jNO config.  Returns ``{}`` if no file is found."""
    global _CONFIG, _CONFIG_PATH
    if _CONFIG is not None and not force:
        return _CONFIG

    for path in _candidate_paths():
        if path.exists():
            _CONFIG = _read_toml(path)
            _CONFIG_PATH = path
            return _CONFIG

    _CONFIG = {}
    _CONFIG_PATH = None
    return _CONFIG


def get_config() -> dict[str, Any]:
    """Return the current (cached) config, loading it first if necessary."""
    if _CONFIG is None:
        load_config()
    return _CONFIG  # type: ignore[return-value]


def get_config_path() -> Path | None:
    """Return the path of the loaded config file, or ``None``."""
    if _CONFIG is None:
        load_config()
    return _CONFIG_PATH


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def get_runs_base_dir() -> str:
    """Return ``runs.base_dir`` from config, defaulting to ``"./runs"``."""
    cfg = get_config()
    return cfg.get("runs", {}).get("base_dir", "./runs")


def get_rsa_public_key() -> str | None:
    """Return the RSA public key path from config, or ``None``."""
    cfg = get_config()
    raw = cfg.get("rsa", {}).get("public_key")
    return os.path.expanduser(raw) if raw else None


def get_rsa_private_key() -> str | None:
    """Return the RSA private key path from config, or ``None``."""
    cfg = get_config()
    raw = cfg.get("rsa", {}).get("private_key")
    return os.path.expanduser(raw) if raw else None


def get_seed() -> int | None:
    """Return ``jno.seed`` from config, or ``None`` if not set."""
    cfg = get_config()
    return cfg.get("jno", {}).get("seed", None)


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------


def setup(script_file: str, name: str | None = None) -> str:
    """Initialise logging and return the run directory for *script_file*.

    Replaces the two-line boilerplate at the top of every example::

        dire = "./runs/heat_equation"
        jno.logger(dire)

    with a single call::

        dire = jno.setup(__file__)

    The run directory is derived as ``<base_dir>/<stem>`` where *stem* is the
    script filename without extension (e.g. ``heat_equation``), and *base_dir*
    comes from ``runs.base_dir`` in the jNO config (default ``"./runs"``).
    Pass an explicit *name* to override the stem.

    A global RNG seed can be set in ``.jno.toml`` under ``[jno] seed`` so that
    all ``jno.core(...)`` instances use it automatically::

        [jno]
        seed = 42

    Args:
        script_file: Pass ``__file__`` from the calling script.
        name: Override the subdirectory name (defaults to the script stem).

    Returns:
        The path of the run directory (created if absent).
    """
    from .logger import init_default_logger

    stem = name or Path(script_file).stem
    dire = str(Path(get_runs_base_dir()) / stem)
    init_default_logger(dire)
    return dire
