"""
Cold Start Experiment
=====================
Runs the same prompt set across all 4 modes starting from empty models.
Tests whether learning modes converge faster and produce lower error than baseline.

Usage:
    python experiment_cold_start.py
"""

import json
import os
import time
from Learning_agent import run_agent

# ── Configuration ────────────────────────────────────────────────────────────

PROMPTS = [
    "I want a spicy Thai curry with chicken",
    "Make me a healthy Mediterranean salad with feta",
    "I want a quick Italian pasta with garlic and basil",
    "Give me a spicy Mexican burrito bowl with beans",
    "I want a light Japanese miso soup with tofu",
]

MODES = ["baseline", "world_model_only", "self_model_only", "full_model"]
ROUNDS = 5  # number of times to repeat the full prompt set
OUTPUT_DIR = "experiments/cold_start"

# ── Run experiment ───────────────────────────────────────────────────────────

def run_cold_start():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mode in MODES:
        wm_path = os.path.join(OUTPUT_DIR, f"world_model_{mode}.json")
        sm_path = os.path.join(OUTPUT_DIR, f"self_model_{mode}.json")
        log_path = os.path.join(OUTPUT_DIR, f"run_log_{mode}.jsonl")

        # Clean slate — no pretraining
        for p in [wm_path, sm_path, log_path]:
            if os.path.exists(p):
                os.remove(p)

        print(f"\n{'='*60}")
        print(f"MODE: {mode}")
        print(f"{'='*60}")

        for round_num in range(ROUNDS):
            for i, prompt in enumerate(PROMPTS):
                run_id = f"round={round_num} prompt={i}"
                try:
                    result = run_agent(
                        user_input=prompt,
                        mode=mode,
                        world_model_path=wm_path,
                        self_model_path=sm_path,
                        log_path=log_path,
                    )
                    print(
                        f"  [{mode}] {run_id} "
                        f"abs_error={result['abs_error']:.3f} "
                        f"predicted={result['predicted_taste']:.3f} "
                        f"actual={result['actual_taste']:.3f} "
                        f"converged={result['convergence_flag']}"
                    )
                except Exception as e:
                    print(f"  [{mode}] {run_id} FAILED: {e}")

                # Small delay to avoid rate limiting
                time.sleep(1)

    print(f"\nExperiment complete. Logs in {OUTPUT_DIR}/")


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze():
    print(f"\n{'='*60}")
    print("COLD START ANALYSIS")
    print(f"{'='*60}\n")

    for mode in MODES:
        log_path = os.path.join(OUTPUT_DIR, f"run_log_{mode}.jsonl")
        if not os.path.exists(log_path):
            print(f"{mode}: no log file found")
            continue

        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not entries:
            print(f"{mode}: no entries")
            continue

        errors = [e["abs_error"] for e in entries]
        mid = len(errors) // 2

        first_half = errors[:mid] if mid > 0 else errors
        second_half = errors[mid:] if mid > 0 else errors

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        avg_total = sum(errors) / len(errors)

        # Check convergence — did it ever flag true?
        converged_at = None
        for i, e in enumerate(entries):
            if e.get("convergence_flag"):
                converged_at = i + 1
                break

        # Self-model trajectory
        sm_first = entries[0].get("updated_self_model", {}) if entries else {}
        sm_last = entries[-1].get("updated_self_model", {}) if entries else {}

        print(f"{mode}:")
        print(f"  runs: {len(entries)}")
        print(f"  avg abs_error:  first_half={avg_first:.3f}  second_half={avg_second:.3f}  delta={avg_first - avg_second:+.3f}")
        print(f"  avg abs_error overall: {avg_total:.3f}")
        print(f"  converged at run: {converged_at or 'never'}")
        print(f"  self_model start: spice={sm_first.get('spice_preference', 'n/a'):.3f}  health={sm_first.get('health_bias', 'n/a'):.3f}")
        print(f"  self_model end:   spice={sm_last.get('spice_preference', 'n/a'):.3f}  health={sm_last.get('health_bias', 'n/a'):.3f}")
        print()


if __name__ == "__main__":
    run_cold_start()
    analyze()
