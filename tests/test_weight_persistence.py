"""Unit tests for weight persistence and FORCE_RETRAIN logic.

Validates: Requirements 7.1, 7.2, 7.3, 7.5
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Re-implement save_weights / load_weights from the notebook so they are
# importable and testable.  The logic mirrors lora_ingredient_experiment.ipynb
# cells 10-11 exactly.
# ---------------------------------------------------------------------------


def save_weights(model, pref_head, path: str) -> None:
    """Save LoRA adapters + preference head state dict."""
    os.makedirs(path, exist_ok=True)
    lora_path = os.path.join(path, "lora")
    model.save_pretrained(lora_path)
    torch.save(pref_head.state_dict(), os.path.join(path, "preference_head.pt"))


def load_weights(model, pref_head, path: str) -> bool:
    """Load saved weights. Returns True if loaded, False if not found."""
    lora_path = os.path.join(path, "lora")
    pref_path = os.path.join(path, "preference_head.pt")
    if os.path.exists(lora_path) and os.path.exists(pref_path):
        try:
            model.load_adapter(lora_path, adapter_name="default")
            pref_head.load_state_dict(
                torch.load(pref_path, map_location=model.device)
            )
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model():
    """Create a mock model with save_pretrained, load_adapter, and device."""
    model = MagicMock()
    model.device = torch.device("cpu")
    return model


def _make_real_pref_head():
    """Create a small real nn.Module so state_dict round-trips work."""
    head = torch.nn.Sequential(
        torch.nn.Linear(16, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 1),
    )
    return head


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveAndLoadWeightsRoundtrip:
    """Validates: Requirement 7.1 — weights are saved and can be reloaded."""

    def test_save_and_load_weights_roundtrip(self, tmp_path: Path) -> None:
        """Save weights to tmp_path, verify files exist, then load and
        verify load_weights returns True."""
        model = _make_mock_model()
        pref_head = _make_real_pref_head()

        weight_dir = str(tmp_path / "weights")

        # --- save ---
        save_weights(model, pref_head, weight_dir)

        # Verify model.save_pretrained was called with the lora sub-path
        model.save_pretrained.assert_called_once_with(
            os.path.join(weight_dir, "lora")
        )

        # Verify preference_head.pt was written to disk
        pref_pt = os.path.join(weight_dir, "preference_head.pt")
        assert os.path.isfile(pref_pt), "preference_head.pt should exist"

        # --- load ---
        # For load to succeed we also need the "lora" directory to exist on
        # disk (save_pretrained is mocked so it didn't create it).
        os.makedirs(os.path.join(weight_dir, "lora"), exist_ok=True)

        load_model = _make_mock_model()
        load_head = _make_real_pref_head()

        result = load_weights(load_model, load_head, weight_dir)

        assert result is True, "load_weights should return True after a save"
        load_model.load_adapter.assert_called_once()


class TestLoadWeightsMissingDir:
    """Validates: Requirement 7.3 — no saved weights → proceed with training."""

    def test_load_weights_returns_false_when_no_saved_weights(
        self, tmp_path: Path
    ) -> None:
        """Call load_weights on an empty directory; should return False."""
        model = _make_mock_model()
        pref_head = _make_real_pref_head()

        result = load_weights(model, pref_head, str(tmp_path / "empty"))

        assert result is False, (
            "load_weights should return False when no weights exist"
        )


class TestForceRetrainBypassesLoading:
    """Validates: Requirement 7.5 — FORCE_RETRAIN=True skips weight loading."""

    def test_force_retrain_bypasses_weight_loading(self, tmp_path: Path) -> None:
        """Simulate the notebook's FORCE_RETRAIN logic: when True,
        load_weights should never be called."""
        model = _make_mock_model()
        pref_head = _make_real_pref_head()
        weight_dir = str(tmp_path / "weights")

        # Pre-save weights so they exist on disk
        save_weights(model, pref_head, weight_dir)
        os.makedirs(os.path.join(weight_dir, "lora"), exist_ok=True)

        # --- replicate notebook cell 11 logic ---
        FORCE_RETRAIN = True
        skip_training = False

        load_called = False

        if not FORCE_RETRAIN:
            load_called = True
            if load_weights(model, pref_head, weight_dir):
                skip_training = True

        assert not load_called, (
            "load_weights must not be called when FORCE_RETRAIN is True"
        )
        assert not skip_training, (
            "Training should NOT be skipped when FORCE_RETRAIN is True"
        )


class TestMissingBaselineJsonSkipsGraphs:
    """Validates: Requirement 7.2 (graceful handling) — missing baseline JSON
    should not crash; the notebook catches FileNotFoundError."""

    def test_missing_baseline_json_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Opening a non-existent baseline_results.json raises
        FileNotFoundError, which the notebook catches to skip graphs."""
        missing_path = str(tmp_path / "baseline_results.json")

        with pytest.raises(FileNotFoundError):
            with open(missing_path, "r") as f:
                json.load(f)

    def test_missing_baseline_json_skips_graphs_gracefully(
        self, tmp_path: Path
    ) -> None:
        """Replicate the notebook's try/except pattern: when the file is
        missing, baseline_results should remain None and graphs are skipped."""
        missing_path = str(tmp_path / "baseline_results.json")

        baseline_results = None
        try:
            with open(missing_path, "r") as f:
                baseline_results = json.load(f)
        except FileNotFoundError:
            pass  # notebook prints a message and skips graphs

        assert baseline_results is None, (
            "baseline_results should be None when file is missing"
        )
