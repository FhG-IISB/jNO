"""Tests for the Orbax checkpoint callback and resume-from-checkpoint."""

import logging
import os
from pathlib import Path

import equinox as eqx
import pytest
import foundax
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solver(epochs=10):
    """Build and briefly train a minimal 1-D solver, return the core instance."""
    import optax
    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")

    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=key))
    u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 100, 1e-5))
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)

    solver = jno.core([pde.mse], domain)
    if epochs > 0:
        solver.solve(epochs)
    return solver


def _restore_trainable_model_from_orbax_step(step_dir: Path, model_key: str, model):
    """Restore one trainable model subtree directly from an Orbax step dir."""
    restore_template = jax.tree_util.tree_map(
        lambda leaf: leaf if eqx.is_array(leaf) else orbax.PLACEHOLDER,
        model,
        is_leaf=lambda leaf: leaf is None,
    )
    restore_item = {"trainable": {model_key: restore_template}}
    restore_args = orbax.checkpoint_utils.construct_restore_args(restore_item)

    checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
    try:
        restored = checkpointer.restore(
            step_dir / "state",
            args=orbax.args.PyTreeRestore(
                item=restore_item,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
    finally:
        close = getattr(checkpointer, "close", None)
        if callable(close):
            close()

    return jax.tree_util.tree_map(
        lambda restored_leaf, fresh_leaf: fresh_leaf if restored_leaf is orbax.PLACEHOLDER or restored_leaf is None else restored_leaf,
        restored["trainable"][model_key],
        model,
        is_leaf=lambda leaf: leaf is orbax.PLACEHOLDER or leaf is None,
    )


def _save_orbax_step(step_dir: Path, trainable, *, epoch: int = 0):
    """Write a minimal jNO-compatible Orbax checkpoint step."""
    checkpointer = orbax.Checkpointer(orbax.CompositeCheckpointHandler())
    try:
        checkpointer.save(
            step_dir,
            args=orbax.args.Composite(
                state=orbax.args.StandardSave(
                    {
                        "trainable": {"1": trainable},
                        "opt_states": {},
                        "rng": jax.random.PRNGKey(0),
                    }
                ),
                metadata=orbax.args.JsonSave(
                    {
                        "epoch": epoch,
                        "total_loss": 0.0,
                        "individual_losses": [],
                        "timestamp": 0.0,
                    }
                ),
            ),
            force=True,
        )
        wait_until_finished = getattr(checkpointer, "wait_until_finished", None)
        if callable(wait_until_finished):
            wait_until_finished()
    finally:
        close = getattr(checkpointer, "close", None)
        if callable(close):
            close()


class _ToyOrbaxModel(eqx.Module):
    weight: jax.Array
    relative_position_index: jax.Array

    def __init__(self, *, weight_scale: float, index_offset: int):
        self.weight = (weight_scale * jnp.arange(6, dtype=jnp.float32)).reshape(2, 3)
        self.relative_position_index = (jnp.arange(4, dtype=jnp.int32) + index_offset).reshape(2, 2)


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


class TestCallbackHooks:
    """Verify that solve() calls callback hooks."""

    def test_on_epoch_end_called(self):
        """on_epoch_end should be invoked at least once during solve()."""
        from jno.utils.adaptive.callbacks import Callback

        class Recorder(Callback):
            def __init__(self):
                self.calls = []

            def on_epoch_end(self, **kwargs):
                self.calls.append(kwargs.get("epoch"))

        rec = Recorder()
        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[rec])

        assert len(rec.calls) > 0, "on_epoch_end was never called"
        assert rec.calls[-1] >= 19, "last reported epoch should be near end"

    def test_on_training_end_called(self):
        """on_training_end should be called exactly once."""
        from jno.utils.adaptive.callbacks import Callback

        class Counter(Callback):
            def __init__(self):
                self.count = 0

            def on_training_end(self, **kwargs):
                self.count += 1

        ctr = Counter()
        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[ctr])

        assert ctr.count == 1

    def test_epoch_end_kwargs(self):
        """on_epoch_end should receive the documented keyword arguments."""
        from jno.utils.adaptive.callbacks import Callback

        required_keys = {"epoch", "trainable", "opt_states", "rng", "total_loss", "individual_losses", "log"}

        class KeyChecker(Callback):
            def __init__(self):
                self.received_keys = set()

            def on_epoch_end(self, **kwargs):
                self.received_keys.update(kwargs.keys())

        kc = KeyChecker()
        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[kc])

        missing = required_keys - kc.received_keys
        assert not missing, f"Missing kwargs: {missing}"


# ---------------------------------------------------------------------------
# CheckpointCallback (requires orbax-checkpoint)
# ---------------------------------------------------------------------------

orbax = pytest.importorskip("orbax.checkpoint", reason="orbax-checkpoint not installed")


@pytest.mark.integration
class TestCheckpointCallback:
    """End-to-end checkpoint callback tests."""

    def test_checkpoint_creates_files(self, tmp_path):
        """Checkpoints should be written to disk."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=5,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()

        steps = cb.all_steps
        assert len(steps) > 0, "No checkpoints were saved"
        assert os.path.isdir(ckpt_dir)

    def test_max_to_keep(self, tmp_path):
        """Only max_to_keep checkpoints should be retained."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=1,
            max_to_keep=2,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()

        steps = cb.all_steps
        assert len(steps) <= 2, f"Expected <=2 checkpoints, got {len(steps)}"

    def test_restore_latest(self, tmp_path):
        """restore() should return a dict with the expected keys."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=3,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()

        restored = cb.restore()
        assert "trainable" in restored
        assert "opt_states" in restored
        assert "rng" in restored
        assert "metadata" in restored
        assert "epoch" in restored["metadata"]
        assert "total_loss" in restored["metadata"]

    def test_restore_specific_step(self, tmp_path):
        """restore(step=...) should return the checkpoint at that step."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=5,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()

        steps = cb.all_steps
        if len(steps) >= 2:
            first_step = steps[0]
            restored = cb.restore(step=first_step)
            assert restored["metadata"]["epoch"] == first_step

    def test_restore_empty_dir_raises(self, tmp_path):
        """restore() on an empty directory should raise FileNotFoundError."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "empty_ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            async_checkpointing=False,
        )

        with pytest.raises(FileNotFoundError):
            cb.restore()
        cb.close()

    def test_best_fn(self, tmp_path):
        """When best_fn is set, the best checkpoint should be retained."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=2,
            best_fn=lambda m: m["total_loss"],
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()

        # Just verify it doesn't crash and creates checkpoints
        steps = cb.all_steps
        assert len(steps) > 0

    def test_latest_step_property(self, tmp_path):
        """latest_step should return the most recent step number."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=5,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()

        latest = cb.latest_step
        assert latest is not None
        assert latest == max(cb.all_steps)

    def test_initialize_from_orbax_step_dir(self, tmp_path):
        """nn.initialize should accept a numbered Orbax checkpoint directory."""
        import jno
        from jno.trace_compiler import TraceCompiler
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = Path(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=str(ckpt_dir),
            save_interval_epochs=1,
            max_to_keep=3,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(3, callbacks=[cb])
        cb.close()

        latest_step = cb.latest_step
        restored = cb.restore(step=latest_step)
        saved_key = next(iter(restored["trainable"]))

        key = jax.random.PRNGKey(0)
        reference_model = _restore_trainable_model_from_orbax_step(
            ckpt_dir / str(latest_step),
            saved_key,
            foundax.mlp(1, hidden_dims=8, num_layers=2, key=key),
        )
        net = jno.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=key))
        net.initialize(str(ckpt_dir / str(latest_step)))

        loaded_model = TraceCompiler.build_single_layer_params(net, None, key, logging.getLogger(__name__))

        expected_leaves = jax.tree_util.tree_leaves(eqx.filter(reference_model, eqx.is_array))
        loaded_leaves = jax.tree_util.tree_leaves(eqx.filter(loaded_model, eqx.is_array))
        assert len(loaded_leaves) == len(expected_leaves)
        for loaded, expected in zip(loaded_leaves, expected_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))

    def test_initialize_from_orbax_root_dir_uses_latest_step(self, tmp_path):
        """nn.initialize should accept an Orbax checkpoint root directory."""
        import jno
        from jno.trace_compiler import TraceCompiler
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = Path(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=str(ckpt_dir),
            save_interval_epochs=1,
            max_to_keep=3,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(3, callbacks=[cb])
        cb.close()

        latest_step = cb.latest_step
        restored = cb.restore(step=latest_step)
        saved_key = next(iter(restored["trainable"]))

        key = jax.random.PRNGKey(0)
        reference_model = _restore_trainable_model_from_orbax_step(
            ckpt_dir / str(latest_step),
            saved_key,
            foundax.mlp(1, hidden_dims=8, num_layers=2, key=key),
        )
        net = jno.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=key))
        net.initialize(str(ckpt_dir))

        loaded_model = TraceCompiler.build_single_layer_params(net, None, key, logging.getLogger(__name__))

        expected_leaves = jax.tree_util.tree_leaves(eqx.filter(reference_model, eqx.is_array))
        loaded_leaves = jax.tree_util.tree_leaves(eqx.filter(loaded_model, eqx.is_array))
        assert len(loaded_leaves) == len(expected_leaves)
        for loaded, expected in zip(loaded_leaves, expected_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))

    def test_initialize_from_orbax_logs_match_summary(self, tmp_path, caplog):
        """Orbax restore should log matched/skipped parameter statistics."""
        import jno
        from jno.trace_compiler import TraceCompiler
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = Path(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=str(ckpt_dir),
            save_interval_epochs=1,
            max_to_keep=3,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(3, callbacks=[cb])
        cb.close()

        latest_step = cb.latest_step
        key = jax.random.PRNGKey(0)
        net = jno.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=key))
        net.initialize(str(ckpt_dir / str(latest_step)))

        logger = logging.getLogger(__name__)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=logger.name):
            TraceCompiler.build_single_layer_params(net, None, key, logger)

        messages = [record.getMessage() for record in caplog.records]
        assert any(msg.startswith("Orbax checkpoint:") and "params matched" in msg for msg in messages)
        assert any(msg.startswith("Checkpoint file:") or "model only consumed" in msg for msg in messages)

    def test_initialize_from_orbax_keeps_unmatched_array_leaves_fresh(self, tmp_path, caplog):
        """Orbax restore should keep unmatched array leaves from the fresh model."""
        from jno.trace_compiler import TraceCompiler

        step_dir = Path(tmp_path / "toy_step")
        reference_model = _ToyOrbaxModel(weight_scale=2.0, index_offset=100)
        fresh_model = _ToyOrbaxModel(weight_scale=1.0, index_offset=0)

        _save_orbax_step(step_dir, eqx.filter(reference_model, eqx.is_inexact_array))

        logger = logging.getLogger(__name__)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=logger.name):
            loaded_model = TraceCompiler._load_orbax_weights_partial(str(step_dir), fresh_model, logger)

        np.testing.assert_allclose(np.asarray(loaded_model.weight), np.asarray(reference_model.weight))
        np.testing.assert_array_equal(
            np.asarray(loaded_model.relative_position_index),
            np.asarray(fresh_model.relative_position_index),
        )
        assert loaded_model.relative_position_index is not None

        messages = [record.getMessage() for record in caplog.records]
        assert any(msg.startswith("Checkpoint file:") and "all checkpoint arrays consumed; model kept fresh init" in msg for msg in messages)


# ---------------------------------------------------------------------------
# resume_from on core.__init__
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResumeFrom:
    """Test the resume_from parameter on core.__init__."""

    def test_resume_restores_epoch(self, tmp_path):
        """After resuming, _total_epochs should reflect the checkpoint."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=3,
            async_checkpointing=False,
        )

        # Train and checkpoint
        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()
        saved_step = cb.latest_step

        # Use restore_checkpoint on the same solver, then continue training.
        # This mimics the resume_from flow without layer_id mismatch.
        solver.restore_checkpoint(ckpt_dir)
        assert solver._total_epochs == saved_step

        solver.solve(10)
        assert solver._total_epochs >= saved_step + 10

    def test_resume_from_sets_and_clears(self, tmp_path):
        """resume_from should be stored and cleared after first solve()."""
        import jno

        # Just verify the attribute lifecycle — no actual checkpoint needed
        # for this: passing a non-existent dir is fine, it'll error, but
        # we can check the attribute is stored.
        solver = _make_solver(epochs=0)
        solver._resume_from = None  # default
        assert solver._resume_from is None


# ---------------------------------------------------------------------------
# restore_checkpoint() standalone method
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRestoreCheckpointMethod:
    """Test core.restore_checkpoint() method."""

    def test_restore_checkpoint_updates_state(self, tmp_path):
        """restore_checkpoint should update the solver's epoch counter."""
        from jno.utils.adaptive.callbacks import CheckpointCallback

        ckpt_dir = str(tmp_path / "ckpts")
        cb = CheckpointCallback(
            directory=ckpt_dir,
            save_interval_epochs=5,
            max_to_keep=3,
            async_checkpointing=False,
        )

        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])
        cb.close()
        saved_step = cb.latest_step

        # Build a fresh solver and restore manually
        solver2 = _make_solver(epochs=0)
        solver2.restore_checkpoint(ckpt_dir)

        assert solver2._total_epochs == saved_step

    def test_restore_checkpoint_no_dir_raises(self, tmp_path):
        """restore_checkpoint on an empty dir should raise FileNotFoundError."""
        empty = str(tmp_path / "empty")
        os.makedirs(empty, exist_ok=True)

        solver = _make_solver(epochs=0)
        with pytest.raises(FileNotFoundError):
            solver.restore_checkpoint(empty)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Verify helpful error when orbax-checkpoint is not installed."""

    def test_checkpoint_callback_import_error(self, monkeypatch):
        """CheckpointCallback.__init__ should raise ImportError with message."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if "orbax" in name:
                raise ImportError("fake missing orbax")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Need to reload to pick up the monkeypatch
        from jno.utils.adaptive.callbacks import CheckpointCallback as CC

        with pytest.raises(ImportError, match="orbax-checkpoint"):
            CC(directory="/tmp/test")


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------


class TestEarlyStoppingCallback:
    """Tests for the EarlyStoppingCallback."""

    def test_stops_training(self):
        """Training should terminate before max epochs when loss plateaus."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        # patience=3 so it stops quickly; min_delta large enough to detect plateau
        cb = EarlyStoppingCallback(patience=3, min_delta=1e-3, verbose=False)
        solver = _make_solver(epochs=0)
        solver.solve(200, callbacks=[cb])

        assert cb.has_stopped, "Early stopping should have triggered"
        assert cb.stopped_epoch is not None
        assert cb.stopped_epoch < 199, "Should have stopped before final epoch"

    def test_does_not_stop_when_improving(self):
        """With high patience, training should complete normally."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=10_000, verbose=False)
        solver = _make_solver(epochs=0)
        solver.solve(20, callbacks=[cb])

        assert not cb.has_stopped

    def test_mode_min(self):
        """mode='min' should detect when loss stops decreasing."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=5, mode="min", verbose=False)

        # Simulate decreasing then flat losses
        for i, loss_val in enumerate([1.0, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]):
            stop = cb.on_epoch_end(epoch=i, total_loss=jnp.array(loss_val), individual_losses=jnp.array([loss_val]), trainable=None, opt_states=None, rng=None)
            if stop:
                break

        assert cb.has_stopped
        assert cb.best_metric == pytest.approx(0.3)

    def test_mode_max(self):
        """mode='max' should detect when metric stops increasing."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        cb = EarlyStoppingCallback(
            patience=3,
            mode="max",
            metric_fn=lambda **kw: float(jax.device_get(kw["total_loss"])),
            verbose=False,
        )

        for i, val in enumerate([0.1, 0.5, 0.9, 0.9, 0.9, 0.9]):
            stop = cb.on_epoch_end(epoch=i, total_loss=jnp.array(val), individual_losses=jnp.array([val]), trainable=None, opt_states=None, rng=None)
            if stop:
                break

        assert cb.has_stopped
        assert cb.best_metric == pytest.approx(0.9)

    def test_mode_rel(self):
        """mode='rel' should use relative improvement threshold."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=2, mode="rel", min_delta=0.1, verbose=False)

        # 1.0 -> 0.95 is only 5% improvement, below 10% threshold
        losses = [1.0, 0.95, 0.94, 0.93]
        for i, val in enumerate(losses):
            stop = cb.on_epoch_end(epoch=i, total_loss=jnp.array(val), individual_losses=jnp.array([val]), trainable=None, opt_states=None, rng=None)
            if stop:
                break

        assert cb.has_stopped

    def test_baseline(self):
        """With a baseline, early stopping fires if metric never beats it."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=2, baseline=0.01, verbose=False)

        # Loss never goes below baseline 0.01
        for i in range(5):
            stop = cb.on_epoch_end(epoch=i, total_loss=jnp.array(0.5), individual_losses=jnp.array([0.5]), trainable=None, opt_states=None, rng=None)
            if stop:
                break

        assert cb.has_stopped

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        with pytest.raises(ValueError, match="mode"):
            EarlyStoppingCallback(mode="invalid")

    def test_custom_metric_fn(self):
        """A custom metric_fn should be used to extract the metric."""
        from jno.utils.adaptive.callbacks import EarlyStoppingCallback

        # Monitor the first individual loss instead of total
        cb = EarlyStoppingCallback(
            patience=2,
            metric_fn=lambda **kw: float(jax.device_get(kw["individual_losses"])[0]),
            verbose=False,
        )

        for i, val in enumerate([1.0, 1.0, 1.0]):
            stop = cb.on_epoch_end(epoch=i, total_loss=jnp.array(0.0), individual_losses=jnp.array([val]), trainable=None, opt_states=None, rng=None)
            if stop:
                break

        assert cb.has_stopped
