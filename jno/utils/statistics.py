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
        - Total loss over time
        - Individual constraint losses over time
        - Learning rate schedule
        - Constraint weights over time

        Args:
            save_path: Optional path to save figure (e.g., "./runs/training.png")
            show: Whether to display the plot (default True)

        Returns:
            matplotlib Figure object

        Example:
            crux.solve(1000, ...)
            crux.solve(1000, ...)
            crux.plot_training_history(save_path="./runs/training.png")
        """

        if not self.training_logs:
            self.log.warning("No training logs available.Run solve() first.")
            return None

        # Aggregate logs across all solve() calls
        all_epochs = []
        all_total_loss = []
        all_individual_losses = []
        all_learning_rates = []
        all_weights = []

        epoch_offset = 0
        for log_dict in self.training_logs:
            epochs = np.array(log_dict.get("epoch", []))
            total_loss = np.array(log_dict.get("total_loss", []))
            losses = np.array(log_dict.get("losses", []))
            lr = np.array(log_dict.get("learning_rate", []))
            weights = np.array(log_dict.get("weights", []))

            all_epochs.append(epochs + epoch_offset)
            all_total_loss.append(total_loss)
            if losses.ndim == 2:  # (n_epochs, n_constraints)
                all_individual_losses.append(losses)
            if lr.size > 0:
                all_learning_rates.append(lr)
            if weights.ndim == 2:  # (n_epochs, n_constraints)
                all_weights.append(weights)

            epoch_offset += len(epochs)

        # Concatenate across all solve() calls
        epochs = np.concatenate(all_epochs) if all_epochs else np.array([])
        total_loss = np.concatenate(all_total_loss) if all_total_loss else np.array([])

        # Determine number of subplots needed
        n_panels = 1  # Always have total loss
        has_individual = len(all_individual_losses) > 0
        has_lr = len(all_learning_rates) > 0
        has_weights = len(all_weights) > 0

        if has_individual:
            n_panels += 1
        if has_lr:
            n_panels += 1
        if has_weights:
            n_panels += 1

        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # Plot total loss
        axes[panel_idx].semilogy(epochs, total_loss, linewidth=2)
        axes[panel_idx].set_xlabel("Epoch")
        axes[panel_idx].set_ylabel("Total Loss")
        axes[panel_idx].set_title("Training Loss")
        axes[panel_idx].grid(True, alpha=0.3)
        panel_idx += 1

        # Plot individual constraint losses
        if has_individual:
            individual_losses = np.concatenate(all_individual_losses, axis=0)
            n_constraints = individual_losses.shape[1]
            for i in range(n_constraints):
                axes[panel_idx].semilogy(epochs, individual_losses[:, i], label=f"Constraint {i}", linewidth=2)
            axes[panel_idx].set_xlabel("Epoch")
            axes[panel_idx].set_ylabel("Loss")
            axes[panel_idx].set_title("Individual Constraint Losses")
            axes[panel_idx].legend()
            axes[panel_idx].grid(True, alpha=0.3)
            panel_idx += 1

        # Plot learning rate
        if has_lr:
            learning_rates = np.concatenate(all_learning_rates)
            axes[panel_idx].semilogy(epochs, learning_rates, linewidth=2, color="green")
            axes[panel_idx].set_xlabel("Epoch")
            axes[panel_idx].set_ylabel("Learning Rate")
            axes[panel_idx].set_title("Learning Rate Schedule")
            axes[panel_idx].grid(True, alpha=0.3)
            panel_idx += 1

        # Plot constraint weights
        if has_weights:
            weights = np.concatenate(all_weights, axis=0)
            n_constraints = weights.shape[1]
            for i in range(n_constraints):
                axes[panel_idx].plot(epochs, weights[:, i], label=f"Weight {i}", linewidth=2)
            axes[panel_idx].set_xlabel("Epoch")
            axes[panel_idx].set_ylabel("Weight")
            axes[panel_idx].set_title("Constraint Weights")
            axes[panel_idx].legend()
            axes[panel_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        self.log.info(f"Plot saved to: {path}")
        return self

    def save(self, path: str):
        import cloudpickle

        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

        self.log.info(f"Training statistics were saved to: {path}")

        return self

    @classmethod
    def load(cls, filepath: str) -> "core":
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
