"""Lightweight hardware monitor that runs in a background thread.

Polls GPU (``nvidia-smi``) and CPU/RAM (``psutil``) metrics at a fixed
interval and writes them to a CSV file.  The monitoring thread only calls
``subprocess.run`` (for nvidia-smi) and ``psutil`` — neither touches JAX,
so it cannot interfere with JIT compilation or device execution.

Usage::

    monitor = HardwareMonitor(log_dir="./runs/exp1", interval=2.0)
    monitor.start()
    # ... training ...
    monitor.stop()          # writes summary to logger
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
import threading
from pathlib import Path
from typing import Optional


# ── nvidia-smi helpers ──────────────────────────────────────────────

_GPU_QUERY = "index,name,utilization.gpu,memory.used,memory.total," "temperature.gpu,power.draw"


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
        "ram_pct": vm.percent,
    }


# ── Monitor process target ─────────────────────────────────────────


def _monitor_loop(csv_path: str, interval: float, stop_event):
    """Entry-point for the background process."""
    # Initial psutil call to prime the CPU counter
    try:
        import psutil

        psutil.cpu_percent(interval=None)
    except ImportError:
        pass

    fieldnames: list[str] | None = None
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as fh:
        while not stop_event.is_set():
            ts = time.time()
            cpu_ram = _query_cpu_ram()
            gpus = _query_gpus()

            if not gpus:
                # CPU-only row
                row = {"timestamp": ts, **cpu_ram}
                if fieldnames is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                writer.writerow(row)
            else:
                for g in gpus:
                    row = {"timestamp": ts, **cpu_ram, **g}
                    if fieldnames is None:
                        fieldnames = list(row.keys())
                        writer = csv.DictWriter(fh, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                            write_header = False
                    writer.writerow(row)

            fh.flush()
            # Sleep in small increments so stop_event is checked promptly
            elapsed = time.time() - ts
            remaining = max(0.0, interval - elapsed)
            stop_event.wait(timeout=remaining)


# ── Public API ──────────────────────────────────────────────────────


class HardwareMonitor:
    """Start a background daemon thread that logs hardware metrics to CSV.

    The thread only calls ``subprocess.run`` (nvidia-smi) and ``psutil``,
    so it never touches JAX and cannot interfere with JIT or device ops.

    Args:
        log_dir: Directory for the CSV file (same as ``logger.path``).
        interval: Seconds between samples (default 2).
        filename: CSV filename (default ``hardware_monitor.csv``).
    """

    def __init__(
        self,
        log_dir: str | Path = "./",
        interval: float = 2.0,
        filename: str = "hardware_monitor.csv",
    ):
        self.log_dir = Path(log_dir)
        self.interval = interval
        self.csv_path = self.log_dir / filename
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._start_time: Optional[float] = None

    def start(self):
        """Start the monitoring thread."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=_monitor_loop,
            args=(str(self.csv_path), self.interval, self._stop_event),
            daemon=True,
        )
        self._start_time = time.time()
        self._thread.start()

    def stop(self, logger=None):
        """Stop the monitoring thread and optionally log a summary.

        Args:
            logger: If provided, a short summary of peak usage is logged.
        """
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

        duration = time.time() - self._start_time if self._start_time else 0

        if logger is not None:
            summary = self._summarize()
            if summary:
                logger.info(f"Hardware monitor ({duration:.0f}s, {self.csv_path.name}): {summary}")

    def _summarize(self) -> str:
        """Read the CSV and produce a one-line peak-usage summary."""
        if not self.csv_path.exists():
            return ""

        peak_gpu_util = 0.0
        peak_gpu_mem = 0.0
        gpu_mem_total = 0.0
        peak_cpu = 0.0
        peak_ram = 0.0
        ram_total = 0.0
        n_rows = 0

        try:
            with open(self.csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    n_rows += 1
                    if "gpu_util_pct" in row:
                        peak_gpu_util = max(peak_gpu_util, float(row["gpu_util_pct"]))
                    if "gpu_mem_used_mb" in row:
                        peak_gpu_mem = max(peak_gpu_mem, float(row["gpu_mem_used_mb"]))
                    if "gpu_mem_total_mb" in row:
                        gpu_mem_total = max(gpu_mem_total, float(row["gpu_mem_total_mb"]))
                    if "cpu_pct" in row:
                        peak_cpu = max(peak_cpu, float(row["cpu_pct"]))
                    if "ram_used_gb" in row:
                        peak_ram = max(peak_ram, float(row["ram_used_gb"]))
                    if "ram_total_gb" in row:
                        ram_total = max(ram_total, float(row["ram_total_gb"]))
        except Exception:
            return ""

        parts = []
        if peak_gpu_mem > 0:
            parts.append(f"GPU peak {peak_gpu_mem:.0f}/{gpu_mem_total:.0f} MB " f"({peak_gpu_util:.0f}% util)")
        if peak_ram > 0:
            parts.append(f"RAM peak {peak_ram:.1f}/{ram_total:.1f} GB")
        if peak_cpu > 0:
            parts.append(f"CPU peak {peak_cpu:.0f}%")
        parts.append(f"{n_rows} samples")

        return " | ".join(parts)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
