"""Tests for run_agent orchestrator and convergence detection (tasks 9.1, 9.2)."""

import json
import os
import pytest


# ── Import the functions under test ──────────────────────────────────────────
from Learning_agent import detect_convergence, run_agent


# ── Task 9.1: learning_rate validation ───────────────────────────────────────


class TestRunAgentLearningRateValidation:
    """run_agent must reject learning_rate outside [0.01, 0.1]."""

    def test_learning_rate_too_low(self):
        with pytest.raises(ValueError, match="learning_rate"):
            run_agent("make me pasta", learning_rate=0.001)

    def test_learning_rate_too_high(self):
        with pytest.raises(ValueError, match="learning_rate"):
            run_agent("make me pasta", learning_rate=0.5)

    def test_learning_rate_zero(self):
        with pytest.raises(ValueError, match="learning_rate"):
            run_agent("make me pasta", learning_rate=0.0)

    def test_learning_rate_negative(self):
        with pytest.raises(ValueError, match="learning_rate"):
            run_agent("make me pasta", learning_rate=-0.05)

    def test_learning_rate_boundary_low_accepted(self):
        """0.01 is valid — should NOT raise ValueError for learning_rate."""
        try:
            run_agent("make me pasta", learning_rate=0.01)
        except ValueError as e:
            if "learning_rate" in str(e):
                pytest.fail("learning_rate=0.01 should be accepted")
        except Exception:
            # Any non-ValueError exception (e.g. LLM/network) is fine —
            # it means learning_rate validation passed.
            pass

    def test_learning_rate_boundary_high_accepted(self):
        """0.1 is valid — should NOT raise ValueError for learning_rate."""
        try:
            run_agent("make me pasta", learning_rate=0.1)
        except ValueError as e:
            if "learning_rate" in str(e):
                pytest.fail("learning_rate=0.1 should be accepted")
        except Exception:
            pass


class TestRunAgentModeValidation:
    """run_agent delegates to select_mode, which raises ValueError on bad mode."""

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            run_agent("make me pasta", mode="turbo")


# ── Task 9.2: convergence detection ─────────────────────────────────────────


class TestDetectConvergence:
    """detect_convergence reads the last 5 log entries and checks SelfModel stability."""

    def _write_log(self, tmp_path, entries):
        log_path = str(tmp_path / "run_log.jsonl")
        with open(log_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return log_path

    def _make_sm(self, sp=0.5, hb=0.5, ca=None):
        return {
            "spice_preference": sp,
            "health_bias": hb,
            "cuisine_affinity": ca or {},
        }

    def test_no_file_returns_false(self, tmp_path):
        assert detect_convergence(str(tmp_path / "nonexistent.jsonl")) is False

    def test_fewer_than_5_entries_returns_false(self, tmp_path):
        entries = [
            {"updated_self_model": self._make_sm()} for _ in range(4)
        ]
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is False

    def test_5_identical_entries_returns_true(self, tmp_path):
        sm = self._make_sm(0.6, 0.7, {"italian": 0.3})
        entries = [{"updated_self_model": sm} for _ in range(5)]
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is True

    def test_5_entries_with_tiny_changes_returns_true(self, tmp_path):
        """Changes < 0.001 should still count as converged."""
        entries = []
        base_sp = 0.5
        for i in range(5):
            entries.append({
                "updated_self_model": self._make_sm(
                    sp=base_sp + i * 0.0001,
                    hb=0.5 + i * 0.0002,
                    ca={"asian": 0.3 + i * 0.00005},
                )
            })
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is True

    def test_5_entries_with_large_spice_change_returns_false(self, tmp_path):
        entries = []
        for i in range(5):
            entries.append({
                "updated_self_model": self._make_sm(
                    sp=0.5 + i * 0.01,  # 0.01 change per step >= 0.001
                    hb=0.5,
                    ca={},
                )
            })
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is False

    def test_5_entries_with_large_health_bias_change_returns_false(self, tmp_path):
        entries = []
        for i in range(5):
            entries.append({
                "updated_self_model": self._make_sm(
                    sp=0.5,
                    hb=0.5 + i * 0.005,
                    ca={},
                )
            })
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is False

    def test_5_entries_with_large_cuisine_change_returns_false(self, tmp_path):
        entries = []
        for i in range(5):
            entries.append({
                "updated_self_model": self._make_sm(
                    sp=0.5,
                    hb=0.5,
                    ca={"mexican": 0.1 + i * 0.01},
                )
            })
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is False

    def test_new_cuisine_appearing_returns_false(self, tmp_path):
        """A cuisine key appearing in entry 4 but not entry 3 means a change from 0.0."""
        entries = [
            {"updated_self_model": self._make_sm(ca={})} for _ in range(4)
        ]
        entries.append({
            "updated_self_model": self._make_sm(ca={"thai": 0.05})
        })
        log_path = self._write_log(tmp_path, entries)
        assert detect_convergence(log_path) is False

    def test_more_than_5_entries_uses_last_5(self, tmp_path):
        """Only the last 5 entries matter. First 3 have big changes, last 5 are stable."""
        unstable = [
            {"updated_self_model": self._make_sm(sp=0.1 * i)} for i in range(3)
        ]
        stable_sm = self._make_sm(0.6, 0.7, {"italian": 0.3})
        stable = [{"updated_self_model": stable_sm} for _ in range(5)]
        log_path = self._write_log(tmp_path, unstable + stable)
        assert detect_convergence(log_path) is True
