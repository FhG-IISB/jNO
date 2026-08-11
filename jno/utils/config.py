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
import sys
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


def apply_ad_mode_defaults(diff_type: str | None = None, hessian_type: str | None = None) -> None:
    """Apply AD mode globals from explicit kwargs, falling back to ``[jno]`` TOML.

    Precedence: explicit kwarg > ``[jno] diff_type`` / ``[jno] hessian_type`` >
    historical defaults already set in :mod:`jno.utils.ad_mode`.
    """
    from . import ad_mode as _ad_mode

    cfg = get_config().get("jno", {})
    diff = diff_type if diff_type is not None else cfg.get("diff_type")
    hess = hessian_type if hessian_type is not None else cfg.get("hessian_type")
    if diff is not None:
        _ad_mode.set_ad_mode(diff)
    if hess is not None:
        _ad_mode.set_hessian_mode(hess)


def _log_setup_info(log, script_path: Path, dire: Path, stem: str, wandb_arg) -> None:
    """Log environment and configuration at the start of a run."""
    import jax

    try:
        from importlib.metadata import version as _pkg_version

        jno_ver = _pkg_version("jax-numerical-operators")
    except Exception:
        jno_ver = "unknown"

    from . import ad_mode as _ad_mode

    devs = jax.devices()
    seed = get_seed()
    diff_type = _ad_mode.get_ad_mode()
    hessian_type = _ad_mode.get_hessian_mode()

    log.info(f"jNO {jno_ver}")
    log.info(f"Python {sys.version.split()[0]}")
    log.info(f"JAX {jax.__version__}  backend={jax.default_backend()}")
    log.info(f"Devices ({len(devs)}): {[str(d) for d in devs]}")
    log.info(f"Script: {script_path}")
    log.info(f"Run directory: {dire}")
    log.info(f"RNG seed: {seed}")
    log.info(f"diff_type: {diff_type}")
    log.info(f"hessian_type: {hessian_type}")
    if wandb_arg is not False:
        log.info(f"Weights & Biases: enabled (project={stem})")
    else:
        log.info("Weights & Biases: disabled")

    if _WANDB_RUN is not None:
        _WANDB_RUN.config.update(
            {
                "jno_version": jno_ver,
                "python_version": sys.version.split()[0],
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "num_devices": len(devs),
                "script": str(script_path),
                "run_directory": str(dire),
                "seed": seed,
                "diff_type": diff_type,
                "hessian_type": hessian_type,
            },
            allow_val_change=True,
        )


def enable_compile_cache(directory: str | None = None) -> str:
    """Turn on JAX's cross-process persistent compilation cache and return its directory.

    jNO assembles a fresh XLA program per problem *structure*, and that cost barely moves with mesh
    size: a mixed-order Stokes build measured 161 compilations taking 7.25 s of a 9.79 s assembly,
    and a 15x larger Poisson problem still compiled only 121 programs. Because the cost is fixed
    per structure rather than per DOF, a cache that survives process exit removes almost all of it
    -- the same Stokes assembly measured **9.43 s cold, 2.20 s warm (4.3x)** on a second process,
    for 187 entries and 7.2 MB.

    Off by default because a library should not write to a user's disk uninvited; this is the
    opt-in, also reachable as ``jno.setup(__file__, compile_cache=True)`` or, per project,
    ``[jno] compile_cache = true`` in ``.jno.toml``.

    **The first run is SLOWER, sometimes much slower** -- writing the entries is not free. A 75k-node
    3-D Poisson build measured 7.45 s with no cache, 18.30 s on the run that populates one, and
    5.84 s on every run after. It pays back from the second run onward, so it is worth enabling for
    a repeated workflow (a sweep, an optimisation loop, a test suite, re-running a script) and not
    worth it for a single cold run.

    The win also scales with how much of a build is compilation, which varies a lot by problem: 74%
    for mixed-order Stokes but 41% for a P1 Poisson on a large mesh, where mesh-proportional work
    the cache cannot touch dominates instead.

    Both settings are required. ``jax_compilation_cache_dir`` ALONE DOES NOTHING here: JAX's default
    ``jax_persistent_cache_min_compile_time_secs`` of 1.0 s skips every one of these compilations,
    which are individually sub-second.

    Parameters
    ----------
    directory : cache location. ``None`` uses ``$JNO_COMPILE_CACHE_DIR`` if set, else
        ``~/.cache/jno/xla``.
    """
    import jax

    path = directory or os.getenv("JNO_COMPILE_CACHE_DIR") or os.path.join("~", ".cache", "jno", "xla")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    return path


def setup(
    script_file: str,
    name: str | None = None,
    wandb: bool | dict = False,
    diff_type: str | None = None,
    hessian_type: str | None = None,
    compile_cache: bool | str | None = None,
) -> str:
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

    A global RNG seed and the default AD modes can be set in ``.jno.toml``
    under ``[jno]`` so that all ``jno.core(...)`` instances use them
    automatically::

        [jno]
        seed         = 42
        diff_type    = "forward"        # first-order AD default
        hessian_type = "fwd-over-rev"   # second-order AD default

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
        diff_type: Global default for first-order AD on ``.d`` / ``.diff`` /
            ``d/dt``. One of ``"forward"`` / ``"reverse"``. ``None`` reads
            ``[jno] diff_type`` from the TOML config, or keeps the historical
            default (``"reverse"``).
        hessian_type: Global default for second-order AD on ``.laplacian`` /
            ``.hessian`` / ``.d2`` / ``.dd``. One of ``"fwd-over-rev"``,
            ``"fwd-over-fwd"``, ``"rev-over-rev"``, ``"rev-over-fwd"``.
            ``None`` reads ``[jno] hessian_type`` from the TOML config, or
            keeps the historical default (``"fwd-over-rev"``).
        compile_cache: Persist XLA compilations across processes — see
            :func:`enable_compile_cache`, which this delegates to. Measured 4.3x on a
            mixed-order Stokes assembly (9.43 s cold, 2.20 s warm), because jNO's compilation
            cost is fixed per problem *structure* rather than per DOF.

            * ``None`` (default) — read ``[jno] compile_cache`` from the TOML config; off if absent.
            * ``False`` — off. ``True`` — on, at ``~/.cache/jno/xla``.
            * ``str`` — on, at that directory.

            **Off by default, deliberately: a library should not write to a user's disk uninvited.**
            Worth turning on for anything run more than once — a sweep, an optimisation loop, a test
            suite, or just re-running a script after an edit. Measured on 3-D Poisson at 27,833 nodes:
            first build 4.75 s -> 2.22 s, repeat build 2.48 s -> 1.51 s. Set it once per project with
            ``[jno] compile_cache = true`` in ``.jno.toml`` rather than editing each script.

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

    # Apply AD mode defaults — explicit kwargs win over TOML, which wins
    # over the historical defaults already in ad_mode.py.
    apply_ad_mode_defaults(diff_type, hessian_type)

    # --- Optional persistent XLA compilation cache (explicit kwarg wins over TOML) ---
    cache_opt = get_config().get("jno", {}).get("compile_cache", False) if compile_cache is None else compile_cache
    if cache_opt:
        cache_dir = enable_compile_cache(cache_opt if isinstance(cache_opt, str) else None)
        _logger_mod._default_logger.info(f"XLA compilation cache enabled at {cache_dir}")

    # --- Optional Weights & Biases ---
    _init_wandb(wandb, stem, str(dire))

    _log_setup_info(_logger_mod._default_logger, script_path, dire, stem, wandb)

    return str(dire)


# ---------------------------------------------------------------------------
# Weights & Biases integration
# ---------------------------------------------------------------------------


def wandb_finish() -> None:
    """Flush and close the active W&B run (no-op if W&B is not enabled).

    jNO registers this automatically via :mod:`atexit`, so you only need to
    call it explicitly when you want to close the run before the process exits
    (e.g. to start a second run in the same script).
    """
    global _WANDB_RUN
    if _WANDB_RUN is not None:
        try:
            _WANDB_RUN.finish()
        except Exception:
            pass
        _WANDB_RUN = None


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
            "wandb=True was passed to jno.setup() but the 'wandb' package "
            "is not installed.  Install it with:  pip install wandb",
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

    # Ensure metrics are flushed and the run is marked finished when the
    # Python process exits (covers normal exit, sys.exit, and unhandled
    # exceptions).  The atexit handler is a no-op if wandb_finish() was
    # already called explicitly beforehand.
    import atexit

    atexit.register(wandb_finish)

    try:
        import weave  # type: ignore[import-untyped]

        weave.init("armbrul/jNO")
    except ImportError:
        pass


def get_wandb_run() -> Any:
    """Return the active W&B run, or ``None`` if W&B is not enabled."""
    return _WANDB_RUN


def wandb_log(metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log *metrics* to W&B if a run is active (no-op otherwise).

    Multiple ``wandb_log`` calls per epoch (main metrics → trackers →
    callbacks) all forward the same ``step`` value so W&B merges them into a
    single row at ``_step=epoch``. The ``epoch`` key is also stamped into
    every dict so users can pick either ``_step`` or ``epoch`` as the chart
    X axis.
    """
    if _WANDB_RUN is not None:
        if step is not None and "epoch" not in metrics:
            metrics = {**metrics, "epoch": step}
        _WANDB_RUN.log(metrics, step=step)


def wandb_commit(step: int) -> None:
    """Commit the buffered W&B row for *step* (no-op if W&B is not enabled).

    When ``step=`` is passed to :func:`wandb_log`, W&B buffers the row and
    only flushes it when a higher step is seen.  Call this once after all
    per-epoch log calls to make the row visible in the UI immediately.
    """
    if _WANDB_RUN is not None:
        _WANDB_RUN.log({}, step=step, commit=True)


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

        alert_level_enum = getattr(wandb, "AlertLevel", None)
        alert_level = getattr(alert_level_enum, level, getattr(alert_level_enum, "WARN", level))
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
