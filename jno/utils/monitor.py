"""Lightweight hardware monitor that runs in a background thread.

Polls GPU (``nvidia-smi``) and CPU/RAM (``psutil``) metrics at a fixed
interval and appends structured lines to the run's ``log.txt`` file via
``logger.quiet()`` (file-only, no console output).  All hardware data
therefore ends up in the single log alongside training progress —
no separate ``hardware_monitor.csv`` is created.

Usage::

    monitor = HardwareMonitor(logger=log, interval=2.0)
    monitor.start()
    # ... training ...
    monitor.stop()   # writes peak-usage summary line to log
"""

from __future__ import annotations

import subprocess
import time
import threading
from typing import Optional


# ── nvidia-smi helpers ──────────────────────────────────────────────

_GPU_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def _query_gpus() -> list[dict]:
    """Query all GPUs via ``nvidia-smi``.  Returns a list of dicts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "gpu_util_pct": float(parts[2]) if parts[2] not in ("[N/A]", "") else 0.0,
                "gpu_mem_used_mb": float(parts[3]) if parts[3] not in ("[N/A]", "") else 0.0,
                "gpu_mem_total_mb": float(parts[4]) if parts[4] not in ("[N/A]", "") else 0.0,
                "gpu_temp_c": float(parts[5]) if parts[5] not in ("[N/A]", "") else 0.0,
                "gpu_power_w": float(parts[6]) if parts[6] not in ("[N/A]", "") else 0.0,
            }
        )
    return rows


# ── psutil helpers ──────────────────────────────────────────────────


def _query_cpu_ram() -> dict:
    """Return CPU and RAM usage via ``psutil``."""
    try:
        import psutil
    except ImportError:
        return {}

    vm = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_gb": vm.used / (1024**3),
        "ram_total_gb": vm.total / (1024**3),
    }


def _format_hw_line(cpu_ram: dict, gpus: list[dict]) -> str:
    """Format one hardware sample as a compact human-readable string."""
    parts = []
    if "cpu_pct" in cpu_ram:
        parts.append(f"CPU {cpu_ram['cpu_pct']:.0f}%")
    if "ram_used_gb" in cpu_ram:
        parts.append(f"RAM {cpu_ram['ram_used_gb']:.1f}/{cpu_ram['ram_total_gb']:.1f} GB")
    for g in gpus:
        idx = g["gpu_index"]
        parts.append(f"gpu{idx}: util={g['gpu_util_pct']:.0f}%" f" mem={g['gpu_mem_used_mb']:.0f}/{g['gpu_mem_total_mb']:.0f} MB" f" temp={g['gpu_temp_c']:.0f}\u00b0C" f" power={g['gpu_power_w']:.0f}W")
    return "HW | " + " | ".join(parts)


# ── Monitor thread ──────────────────────────────────────────────────


def _monitor_loop(logger, interval: float, stop_event, peak: dict, lock: threading.Lock):
    """Entry-point for the background monitoring thread."""
    try:
        import psutil

        psutil.cpu_percent(interval=None)  # prime the CPU counter
    except ImportError:
        pass

    while not stop_event.is_set():
        ts = time.time()
        cpu_ram = _query_cpu_ram()
        gpus = _query_gpus()

        # Write structured line to log file only (no console output)
        if logger is not None and hasattr(logger, "quiet"):
            logger.quiet(_format_hw_line(cpu_ram, gpus))

        # Update in-memory peak values
        with lock:
            if "cpu_pct" in cpu_ram:
                peak["cpu_pct"] = max(peak.get("cpu_pct", 0.0), cpu_ram["cpu_pct"])
            if "ram_used_gb" in cpu_ram:
                peak["ram_used_gb"] = max(peak.get("ram_used_gb", 0.0), cpu_ram["ram_used_gb"])
                peak["ram_total_gb"] = cpu_ram["ram_total_gb"]
            for g in gpus:
                idx = g["gpu_index"]
                peak[f"gpu{idx}_util"] = max(peak.get(f"gpu{idx}_util", 0.0), g["gpu_util_pct"])
                peak[f"gpu{idx}_mem"] = max(peak.get(f"gpu{idx}_mem", 0.0), g["gpu_mem_used_mb"])
                peak[f"gpu{idx}_mem_total"] = g["gpu_mem_total_mb"]

        elapsed = time.time() - ts
        stop_event.wait(timeout=max(0.0, interval - elapsed))


# ── Public API ──────────────────────────────────────────────────────


class HardwareMonitor:
    """Background daemon thread that appends hardware metrics to ``log.txt``.

    All hardware data is written via ``logger.quiet()`` (file-only, no
    console output), so it ends up in the same ``log.txt`` used for all
    other run information — no separate CSV is created.

    Args:
        logger: The run's Logger instance (must support ``.quiet()``).
        interval: Seconds between samples (default 2).
    """

    def __init__(self, logger=None, interval: float = 2.0):
        self._logger = logger
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._start_time: Optional[float] = None
        self._peak: dict = {}
        self._lock = threading.Lock()

    def start(self):
        """Start the monitoring thread."""
        self._stop_event = threading.Event()
        self._peak = {}
        self._thread = threading.Thread(
            target=_monitor_loop,
            args=(self._logger, self.interval, self._stop_event, self._peak, self._lock),
            daemon=True,
        )
        self._start_time = time.time()
        self._thread.start()

    def stop(self, logger=None):
        """Stop the monitoring thread and log a peak-usage summary.

        Args:
            logger: If provided, a short summary is logged as an INFO line
                    (appears both on console and in the log file).
        """
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

        duration = time.time() - self._start_time if self._start_time else 0

        _log = logger or self._logger
        if _log is not None:
            summary = self._summarize()
            if summary:
                _log.info(f"Hardware monitor ({duration:.0f}s): {summary}")

    def _summarize(self) -> str:
        """Return a one-line peak-usage summary from in-memory data."""
        with self._lock:
            peak = dict(self._peak)

        if not peak:
            return ""

        parts = []
        gpu_indices = sorted({int(k.split("_")[0][3:]) for k in peak if k.startswith("gpu")})
        for idx in gpu_indices:
            mem = peak.get(f"gpu{idx}_mem", 0.0)
            mem_total = peak.get(f"gpu{idx}_mem_total", 0.0)
            util = peak.get(f"gpu{idx}_util", 0.0)
            if mem > 0:
                parts.append(f"GPU peak {mem:.0f}/{mem_total:.0f} MB ({util:.0f}% util)")
        if "ram_used_gb" in peak:
            parts.append(f"RAM peak {peak['ram_used_gb']:.1f}/{peak.get('ram_total_gb', 0):.1f} GB")
        if "cpu_pct" in peak:
            parts.append(f"CPU peak {peak['cpu_pct']:.0f}%")

        return " | ".join(parts)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
