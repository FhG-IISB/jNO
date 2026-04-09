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
_WANDB_RUN: Any = None  # wandb.Run or None


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
    """Return run base dir with env override support.

    Precedence:
    1) ``JNO_LOG_DIR`` (environment)
    2) ``runs.base_dir`` (config)
    3) ``"./runs"`` (default)
    """
    env = os.getenv("JNO_LOG_DIR")
    if env:
        return os.path.expanduser(env)

    cfg = get_config()
    return cfg.get("runs", {}).get("base_dir", "./runs")


def get_rsa_public_key() -> str | None:
    """Return RSA public key path with env override support.

    Precedence:
    1) ``JNO_RSA_PUBLIC_KEY`` (environment)
    2) ``JNO_RSA_PATH`` + ``/public.pem`` (environment)
    3) ``rsa.public_key`` (config)
    4) ``None``
    """
    env_pub = os.getenv("JNO_RSA_PUBLIC_KEY")
    if env_pub:
        return os.path.expanduser(env_pub)

    env_rsa_path = os.getenv("JNO_RSA_PATH")
    if env_rsa_path:
        return str((Path(os.path.expanduser(env_rsa_path)) / "public.pem").resolve())

    cfg = get_config()
    raw = cfg.get("rsa", {}).get("public_key")
    return os.path.expanduser(raw) if raw else None


def get_rsa_private_key() -> str | None:
    """Return RSA private key path with env override support.

    Precedence:
    1) ``JNO_RSA_PRIVATE_KEY`` (environment)
    2) ``JNO_RSA_PATH`` + ``/private.pem`` (environment)
    3) ``rsa.private_key`` (config)
    4) ``None``
    """
    env_priv = os.getenv("JNO_RSA_PRIVATE_KEY")
    if env_priv:
        return os.path.expanduser(env_priv)

    env_rsa_path = os.getenv("JNO_RSA_PATH")
    if env_rsa_path:
        return str((Path(os.path.expanduser(env_rsa_path)) / "private.pem").resolve())

    cfg = get_config()
    raw = cfg.get("rsa", {}).get("private_key")
    return os.path.expanduser(raw) if raw else None


def get_seed() -> int:
    """Return seed with env override support.

    Precedence:
    1) ``JNO_SEED`` (environment)
    2) ``jno.seed`` (config)
    3) ``42`` (default)
    """
    env_seed = os.getenv("JNO_SEED")
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError as e:
            raise ValueError(f"Invalid JNO_SEED={env_seed!r}; expected integer.") from e

    cfg = get_config()
    return cfg.get("jno", {}).get("seed", 42)


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------


def setup(script_file: str, name: str | None = None, wandb: bool | dict = False) -> str:
    """Initialise logging and return the run directory for *script_file*.

    Replaces the two-line boilerplate at the top of every example::

        dire = "./runs/heat_equation"
        jno.logger(dire)

    with a single call::

        dire = jno.setup(__file__)

    The run directory is derived as ``<base_dir>/<stem>`` where *stem* is the
    script filename without extension (e.g. ``heat_equation``), and *base_dir*
    comes from ``runs.base_dir`` in the jNO config (default ``"./runs"``).
    Relative ``base_dir`` values are resolved against ``script_file``'s parent
    directory so output paths are stable regardless of current shell cwd.
    Pass an explicit *name* to override the stem.

    A global RNG seed can be set in ``.jno.toml`` under ``[jno] seed`` so that
    all ``jno.core(...)`` instances use it automatically::

        [jno]
        seed = 42

    Args:
        script_file: Pass ``__file__`` from the calling script.
        name: Override the subdirectory name (defaults to the script stem).
        wandb: Enable `Weights & Biases <https://wandb.ai>`_ logging.

            * ``False`` (default) — disabled.
            * ``True`` — call ``wandb.init(project=stem, dir=run_dir)``.
            * ``dict`` — passed as keyword arguments to ``wandb.init()``,
              with *project* and *dir* filled in as defaults if absent.

            Requires the ``wandb`` package to be installed. When the
            import fails and *wandb* is not ``False``, a warning is
            printed and training continues without W&B logging.

    Returns:
        The path of the run directory (created if absent).
    """
    from . import logger as _logger_mod

    script_path = Path(script_file).resolve()
    stem = name or script_path.stem

    base_dir = Path(os.path.expanduser(get_runs_base_dir()))
    if not base_dir.is_absolute():
        base_dir = script_path.parent / base_dir

    dire = (base_dir / stem).resolve()

    # Always bind a fresh concrete logger for this run directory.
    # This avoids stale singleton state where console logging is active
    # but file logging is not attached.
    try:
        old_logger = getattr(_logger_mod, "_default_logger", None)
        if old_logger is not None and hasattr(old_logger, "close"):
            old_logger.close()
    except Exception:
        pass

    _logger_mod._default_logger = _logger_mod.Logger(path=dire, log_print=(True, True), name="DefaultLogger")

    # Seed jno.nn default PRNG stream from config so model factories can
    # omit explicit key=... after setup().
    try:
        from ..architectures.models import set_default_rng_seed

        set_default_rng_seed(get_seed())
    except Exception:
        # Keep setup robust even if architecture modules are unavailable.
        pass

    # --- Optional Weights & Biases ---
    _init_wandb(wandb, stem, str(dire))

    return str(dire)


# ---------------------------------------------------------------------------
# Weights & Biases integration
# ---------------------------------------------------------------------------


def _init_wandb(wandb_arg: bool | dict, project: str, run_dir: str) -> None:
    """Initialise a W&B run based on the *wandb* argument to :func:`setup`."""
    global _WANDB_RUN

    if wandb_arg is False:
        _WANDB_RUN = None
        return

    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        import warnings

        warnings.warn(
            "wandb=True was passed to jno.setup() but the 'wandb' package " "is not installed.  Install it with:  pip install wandb",
            stacklevel=3,
        )
        _WANDB_RUN = None
        return

    kwargs: dict[str, Any] = {}
    if isinstance(wandb_arg, dict):
        kwargs.update(wandb_arg)

    kwargs.setdefault("project", project)
    kwargs.setdefault("dir", run_dir)

    _WANDB_RUN = wandb.init(**kwargs)
    _WANDB_RUN.log_code()


def get_wandb_run() -> Any:
    """Return the active W&B run, or ``None`` if W&B is not enabled."""
    return _WANDB_RUN


def wandb_log(metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log *metrics* to W&B if a run is active (no-op otherwise)."""
    if _WANDB_RUN is not None:
        _WANDB_RUN.log(metrics, step=step)


def wandb_alert(title: str, text: str, level: str = "WARN") -> None:
    """Send a W&B alert if a run is active (no-op otherwise).

    Args:
        title: Short alert title.
        text: Alert body text.
        level: One of ``"INFO"``, ``"WARN"``, or ``"ERROR"``.
    """
    if _WANDB_RUN is None:
        return
    try:
        import wandb  # type: ignore[import-untyped]

        alert_level = getattr(wandb.AlertLevel, level, wandb.AlertLevel.WARN)
        _WANDB_RUN.alert(title=title, text=text, level=alert_level)
    except Exception:
        pass


def wandb_log_model(solver: Any, name: str = "model") -> None:
    """Upload the trained *solver* as a versioned W&B artifact.

    Serialises *solver* with ``cloudpickle`` to a temporary file and
    logs it as a ``model`` artifact.  No-op when W&B is not active.

    Args:
        solver: A :class:`~jno.core.core` instance (or any picklable object).
        name: Artifact name.  Defaults to ``"model"``.
    """
    if _WANDB_RUN is None:
        return

    import tempfile
    import cloudpickle

    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"{name}.pkl")
        with open(path, "wb") as f:
            cloudpickle.dump(solver, f)
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        _WANDB_RUN.log_artifact(artifact)
