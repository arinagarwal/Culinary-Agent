"""Unit tests for log_run (RunLogger) — validates Requirements 10.1–10.4, 13.1–13.3."""

import json
import os
import sys

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Learning_agent import log_run


REQUIRED_KEYS = [
    "run_number",
    "mode",
    "recipe",
    "predicted_taste",
    "actual_taste",
    "error",
    "abs_error",
    "learning_rate",
    "updated_world_model",
    "updated_self_model",
    "convergence_flag",
]


def _make_record(run_number=1, mode="full_model", error=0.17):
    """Return a minimal but complete run record."""
    return {
        "run_number": run_number,
        "mode": mode,
        "recipe": {"recipe_name": "test", "ingredients_required": []},
        "predicted_taste": 0.45,
        "actual_taste": 0.45 + error,
        "error": error,
        "abs_error": abs(error),
        "learning_rate": 0.05,
        "updated_world_model": {"ingredient_weights": {}},
        "updated_self_model": {"spice_preference": 0.5, "health_bias": 0.5, "cuisine_affinity": {}},
        "convergence_flag": False,
    }


def test_creates_file_if_missing(tmp_path):
    """Req 10.4 — log file is created on first call."""
    log_file = str(tmp_path / "run_log.jsonl")
    assert not os.path.exists(log_file)
    log_run(log_file, _make_record())
    assert os.path.exists(log_file)


def test_appends_single_json_line(tmp_path):
    """Req 10.2, 10.3 — each call appends exactly one valid JSON line."""
    log_file = str(tmp_path / "run_log.jsonl")
    log_run(log_file, _make_record(run_number=1))
    log_run(log_file, _make_record(run_number=2))

    with open(log_file) as f:
        lines = f.read().splitlines()

    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)  # must not raise
        assert isinstance(parsed, dict)


def test_record_contains_all_required_keys(tmp_path):
    """Req 10.1, 13.1, 13.2, 13.3 — every required field is present."""
    log_file = str(tmp_path / "run_log.jsonl")
    record = _make_record()
    log_run(log_file, record)

    with open(log_file) as f:
        parsed = json.loads(f.readline())

    for key in REQUIRED_KEYS:
        assert key in parsed, f"Missing required key: {key}"


def test_multiple_appends_preserve_order(tmp_path):
    """Req 10.3 — records appear in append order with correct run_numbers."""
    log_file = str(tmp_path / "run_log.jsonl")
    for i in range(1, 6):
        log_run(log_file, _make_record(run_number=i))

    with open(log_file) as f:
        lines = f.read().splitlines()

    assert len(lines) == 5
    for idx, line in enumerate(lines, start=1):
        assert json.loads(line)["run_number"] == idx


def test_abs_error_matches_error(tmp_path):
    """Req 13.2 — abs_error equals abs(error)."""
    log_file = str(tmp_path / "run_log.jsonl")
    record = _make_record(error=-0.23)
    log_run(log_file, record)

    with open(log_file) as f:
        parsed = json.loads(f.readline())

    assert parsed["abs_error"] == abs(parsed["error"])
