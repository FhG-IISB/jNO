import matplotlib.pyplot as plt
import numpy as np
from .logger import get_logger


class statistics:

    def __init__(self, logs):
        self.training_logs = logs
        self.log = get_logger()

    def plot(self, path: str = None):
        """Plot training statistics from all solve() calls.

        Creates a multi-panel figure showing:
        - Constraint losses over time (individual lines; total added when >1 constraint)
        - Tracker values over time (if any trackers were defined)
        - Step time in milliseconds (derived from log timestamps)

        Args:
            path: Path to save the figure (e.g. "./runs/training.png").

        Returns:
            self (for chaining)
        """

        if not self.training_logs:
            self.log.warning("No training logs available. Run solve() first.")
            return None

        # ── Aggregate across all solve() calls ──────────────────────────
        all_epochs = []
        all_total_loss = []
        all_losses = []  # (n_log, n_constraints) per call
        all_timestamps = []
        all_track_stats = []  # (n_track_logs, n_trackers) per call — may be absent

        epoch_offset = 0
        for log_dict in self.training_logs:
            epochs = np.array(log_dict.get("epoch", []))
            total_loss = np.array(log_dict.get("total_loss", []))
            losses = np.array(log_dict.get("losses", []))
            ts = np.array(log_dict.get("timestamps", []))
            track = np.array(log_dict.get("track_stats", []))

            all_epochs.append(epochs + epoch_offset)
            all_total_loss.append(total_loss)
            if losses.ndim == 2:
                all_losses.append(losses)
            if ts.size > 0:
                all_timestamps.append(ts)
            if track.ndim == 2 and track.size > 0:
                all_track_stats.append(track)

            epoch_offset += int(epochs[-1]) + 1 if epochs.size > 0 else 0

        epochs = np.concatenate(all_epochs) if all_epochs else np.array([])
        total_loss = np.concatenate(all_total_loss) if all_total_loss else np.array([])
        losses = np.concatenate(all_losses, axis=0) if all_losses else None
        timestamps = np.concatenate(all_timestamps) if all_timestamps else None
        track_stats = np.concatenate(all_track_stats, axis=0) if all_track_stats else None

        n_constraints = losses.shape[1] if losses is not None else 0
        has_multi = n_constraints > 1
        has_trackers = track_stats is not None
        has_timing = timestamps is not None and len(timestamps) > 1

        # ── Layout ──────────────────────────────────────────────────────
        n_panels = 1 + int(has_multi) + int(has_trackers) + int(has_timing)
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), squeeze=False)
        axes = axes[:, 0]
        ax = iter(axes)

        # ── Panel 1: losses ─────────────────────────────────────────────
        a = next(ax)
        if losses is not None:
            for i in range(n_constraints):
                a.semilogy(epochs, losses[:, i], linewidth=1.8, label=f"C{i}")
            if has_multi:
                a.semilogy(epochs, total_loss, linewidth=2, linestyle="--", color="black", label="total")
            a.legend(fontsize=8)
        else:
            a.semilogy(epochs, total_loss, linewidth=2)
        a.set_xlabel("Epoch")
        a.set_ylabel("Loss")
        a.set_title("Training Loss")
        a.grid(True, alpha=0.3)

        # ── Panel 2 (optional): per-constraint breakdown when >1 ────────
        if has_multi:
            a = next(ax)
            for i in range(n_constraints):
                a.semilogy(epochs, losses[:, i], linewidth=1.8, label=f"Constraint {i}")
            a.set_xlabel("Epoch")
            a.set_ylabel("Loss")
            a.set_title("Individual Constraint Losses")
            a.legend(fontsize=8)
            a.grid(True, alpha=0.3)

        # ── Panel: tracker stats ─────────────────────────────────────────
        if has_trackers:
            a = next(ax)
            # track_stats rows are logged at tracker intervals, not per print_rate,
            # so we just use an index axis if epoch counts don't align
            t_epochs = np.linspace(epochs[0], epochs[-1], len(track_stats)) if len(track_stats) != len(epochs) else epochs
            for i in range(track_stats.shape[1]):
                a.semilogy(t_epochs, track_stats[:, i], linewidth=1.8, label=f"Tracker {i}")
            a.set_xlabel("Epoch")
            a.set_ylabel("Value")
            a.set_title("Tracker Values")
            a.legend(fontsize=8)
            a.grid(True, alpha=0.3)

        # ── Panel: step time ─────────────────────────────────────────────
        if has_timing:
            a = next(ax)
            step_ms = np.diff(timestamps) * 1e3  # seconds → ms
            step_epochs = epochs[1:]  # one fewer point
            a.plot(step_epochs, step_ms, linewidth=1.2, color="steelblue")
            a.axhline(np.median(step_ms), color="tomato", linestyle="--", linewidth=1.2, label=f"median {np.median(step_ms):.1f} ms")
            a.set_xlabel("Epoch")
            a.set_ylabel("Step time (ms)")
            a.set_title("Per-step Wall Time")
            a.legend(fontsize=8)
            a.grid(True, alpha=0.3)

        # ── Footer: summary text ─────────────────────────────────────────
        total_time = sum(d.get("training_time", 0) for d in self.training_logs)
        n_trainable = self.training_logs[-1].get("trainable_params", 0)
        _t = int(total_time)
        fig.text(0.5, -0.01, f"Trainable params: {n_trainable:,}   |   " f"Total training time: {_t // 3600}h {(_t % 3600) // 60}m {_t % 60}s", ha="center", fontsize=9, color="gray")

        plt.tight_layout()
        if path is not None:
            from pathlib import Path as _P

            _P(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.log.info(f"Plot saved to: {path}")
        plt.close(fig)
        return self

    def save(self, path: str):
        import cloudpickle

        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

        self.log.info(f"Training statistics were saved to: {path}")

        return self

    @classmethod
    def load(cls, filepath: str) -> "statistics":  # type: ignore[name-defined]
        """Load a trained core model from a file.

        Restores all trained parameters, operations, domain, and history.

        Args:
            filepath: Path to saved model file

        Returns:
            core instance with trained parameters

        Example:
            sol = core.load("trained_model.pkl")
        """
        import cloudpickle

        with open(filepath, "rb") as f:
            core = cloudpickle.load(f)
        return core
