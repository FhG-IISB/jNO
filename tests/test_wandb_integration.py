"""Tests for the Weights & Biases integration in jno.setup() and core.solve()."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

import jno.utils.config as cfg_module


class TestInitWandb:
    """Unit tests for _init_wandb() and get_wandb_run()."""

    def setup_method(self):
        cfg_module._WANDB_RUN = None

    def teardown_method(self):
        cfg_module._WANDB_RUN = None

    def test_false_disables(self):
        """wandb=False leaves the run as None."""
        cfg_module._init_wandb(False, "proj", "/tmp/run")
        assert cfg_module.get_wandb_run() is None

    def test_true_calls_wandb_init(self):
        """wandb=True calls wandb.init with project and dir defaults."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cfg_module._init_wandb(True, "heat_eq", "/tmp/runs/heat_eq")

        mock_wandb.init.assert_called_once_with(
            project="heat_eq", dir="/tmp/runs/heat_eq"
        )
        assert cfg_module.get_wandb_run() is mock_run

    def test_dict_passes_kwargs(self):
        """wandb=dict forwards kwargs and fills defaults."""
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cfg_module._init_wandb(
                {"tags": ["pinn"], "group": "sweep"},
                "heat_eq",
                "/tmp/runs/heat_eq",
            )

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "heat_eq"
        assert call_kwargs["dir"] == "/tmp/runs/heat_eq"
        assert call_kwargs["tags"] == ["pinn"]
        assert call_kwargs["group"] == "sweep"

    def test_dict_respects_user_project(self):
        """User-supplied project in dict is not overridden."""
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cfg_module._init_wandb(
                {"project": "custom"},
                "heat_eq",
                "/tmp/runs/heat_eq",
            )

        assert mock_wandb.init.call_args[1]["project"] == "custom"

    def test_missing_wandb_warns(self):
        """When wandb is not installed, a warning is issued."""
        import sys

        saved = sys.modules.get("wandb")
        sys.modules["wandb"] = None  # simulate ImportError

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cfg_module._init_wandb(True, "proj", "/tmp")

            assert cfg_module.get_wandb_run() is None
            assert any("wandb" in str(x.message) for x in w)
        finally:
            if saved is not None:
                sys.modules["wandb"] = saved
            else:
                sys.modules.pop("wandb", None)


class TestWandbLog:
    """Unit tests for wandb_log()."""

    def setup_method(self):
        cfg_module._WANDB_RUN = None

    def teardown_method(self):
        cfg_module._WANDB_RUN = None

    def test_noop_when_no_run(self):
        """wandb_log does nothing when no run is active."""
        cfg_module.wandb_log({"loss": 1.0}, step=0)  # should not raise

    def test_delegates_to_run(self):
        """wandb_log forwards to the active run's .log()."""
        mock_run = MagicMock()
        cfg_module._WANDB_RUN = mock_run

        cfg_module.wandb_log({"loss": 0.5}, step=42)

        mock_run.log.assert_called_once_with({"loss": 0.5}, step=42)


class TestWandbLogModel:
    """Unit tests for wandb_log_model()."""

    def setup_method(self):
        cfg_module._WANDB_RUN = None

    def teardown_method(self):
        cfg_module._WANDB_RUN = None

    def test_noop_when_no_run(self):
        """wandb_log_model does nothing when no run is active."""
        cfg_module.wandb_log_model({"dummy": "object"})  # should not raise

    def test_uploads_artifact(self):
        """wandb_log_model serialises and uploads a model artifact."""
        mock_run = MagicMock()
        cfg_module._WANDB_RUN = mock_run

        mock_artifact = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cfg_module.wandb_log_model({"weights": [1, 2, 3]}, name="my_model")

        mock_wandb.Artifact.assert_called_once_with("my_model", type="model")
        mock_artifact.add_file.assert_called_once()
        mock_run.log_artifact.assert_called_once_with(mock_artifact)


class TestWandbAlert:
    """Unit tests for wandb_alert()."""

    def setup_method(self):
        cfg_module._WANDB_RUN = None

    def teardown_method(self):
        cfg_module._WANDB_RUN = None

    def test_noop_when_no_run(self):
        """wandb_alert does nothing when no run is active."""
        cfg_module.wandb_alert("title", "text")  # should not raise

    def test_delegates_to_run(self):
        """wandb_alert forwards to the active run's .alert()."""
        mock_run = MagicMock()
        cfg_module._WANDB_RUN = mock_run

        mock_wandb = MagicMock()
        mock_wandb.AlertLevel.WARN = "WARN_SENTINEL"

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cfg_module.wandb_alert("NaN loss", "Exploded at epoch 5", level="WARN")

        mock_run.alert.assert_called_once_with(
            title="NaN loss",
            text="Exploded at epoch 5",
            level="WARN_SENTINEL",
        )

    def test_error_level(self):
        """wandb_alert correctly maps the ERROR level."""
        mock_run = MagicMock()
        cfg_module._WANDB_RUN = mock_run

        mock_wandb = MagicMock()
        mock_wandb.AlertLevel.ERROR = "ERROR_SENTINEL"

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cfg_module.wandb_alert("Bad", "Really bad", level="ERROR")

        mock_run.alert.assert_called_once_with(
            title="Bad",
            text="Really bad",
            level="ERROR_SENTINEL",
        )
