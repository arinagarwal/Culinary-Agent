"""
Warm Start Experiment
=====================
Phase 1: Pretrain models with a diverse set of warmup prompts.
Phase 2: Snapshot the pretrained models, then run the same evaluation
         prompt set across all 4 modes starting from that snapshot.

Tests whether a pretrained agent maintains more consistent preferences
than baseline.

Usage:
    python experiment_warm_start.py
"""

import json
import os
import shutil
import time
from Learning_agent import run_agent

# ── Configuration ────────────────────────────────────────────────────────────

WARMUP_PROMPTS = [
    "I want a creamy mushroom risotto",
    "Make me a spicy Korean bibimbap",
    "I want a fresh Greek salad with olives",
    "Give me a rich French onion soup",
    "I want a tangy Indian chicken tikka masala",
    "Make me a simple American cheeseburger",
    "I want a light Vietnamese pho with herbs",
    "Give me a hearty Irish beef stew",
    "I want a sweet Japanese teriyaki salmon",
    "Make me a smoky Texan BBQ pulled pork sandwich",
    "I want a refreshing Brazilian acai bowl",
    "Give me a classic British fish and chips",
    "I want a fragrant Moroccan tagine with lamb",
    "Make me a crispy Chinese kung pao chicken",
    "I want a comforting Italian minestrone soup",
    "Give me a zesty Peruvian ceviche",
    "I want a warm Ethiopian lentil stew",
    "Make me a savory Turkish kebab plate",
    "I want a tropical Hawaiian poke bowl",
    "Give me a rustic Spanish paella with seafood",
]

EVAL_PROMPTS = [
    "I want a spicy Thai curry with chicken",
    "Make me a healthy Mediterranean salad with feta",
    "I want a quick Italian pasta with garlic and basil",
    "Give me a spicy Mexican burrito bowl with beans",
    "I want a light Japanese miso soup with tofu",
]

MODES = ["baseline", "world_model_only", "self_model_only", "full_model"]
EVAL_ROUNDS = 5
OUTPUT_DIR = "experiments/warm_start"
PRETRAIN_DIR = os.path.join(OUTPUT_DIR, "pretrained")

# ── Phase 1: Pretraining ────────────────────────────────────────────────────

def pretrain():
    os.makedirs(PRETRAIN_DIR, exist_ok=True)

    wm_path = os.path.join(PRETRAIN_DIR, "world_model.json")
    sm_path = os.path.join(PRETRAIN_DIR, "self_model.json")
    log_path = os.path.join(PRETRAIN_DIR, "warmup_log.jsonl")

    # Clean slate for pretraining
    for p in [wm_path, sm_path, log_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"{'='*60}")
    print("PHASE 1: PRETRAINING (full_model mode)")
    print(f"{'='*60}")

    for i, prompt in enumerate(WARMUP_PROMPTS):
        try:
            result = run_agent(
                user_input=prompt,
                mode="full_model",
                world_model_path=wm_path,
                self_model_path=sm_path,
                log_path=log_path,
            )
            print(
                f"  warmup {i+1}/{len(WARMUP_PROMPTS)} "
                f"abs_error={result['abs_error']:.3f} "
                f"converged={result['convergence_flag']}"
            )
        except Exception as e:
            print(f"  warmup {i+1}/{len(WARMUP_PROMPTS)} FAILED: {e}")

        time.sleep(1)

    # Print pretrained model state
    if os.path.exists(sm_path):
        with open(sm_path) as f:
            sm = json.load(f)
        print(f"\nPretrained SelfModel:")
        print(f"  spice_preference: {sm.get('spice_preference', 0.5):.3f}")
        print(f"  health_bias: {sm.get('health_bias', 0.5):.3f}")
        print(f"  cuisine_affinity: {sm.get('cuisine_affinity', {})}")

    if os.path.exists(wm_path):
        with open(wm_path) as f:
            wm = json.load(f)
        weights = wm.get("ingredient_weights", {})
        top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nPretrained WorldModel top 10 ingredients:")
        for name, w in top:
            print(f"  {name}: {w:.4f}")

    print(f"\nPretraining complete. Snapshot in {PRETRAIN_DIR}/")


# ── Phase 2: Evaluation ─────────────────────────────────────────────────────

def evaluate():
    print(f"\n{'='*60}")
    print("PHASE 2: EVALUATION (from pretrained snapshot)")
    print(f"{'='*60}")

    pretrained_wm = os.path.join(PRETRAIN_DIR, "world_model.json")
    pretrained_sm = os.path.join(PRETRAIN_DIR, "self_model.json")

    if not os.path.exists(pretrained_wm) or not os.path.exists(pretrained_sm):
        print("ERROR: Pretrained models not found. Run pretrain() first.")
        return

    for mode in MODES:
        # Copy pretrained models as starting point for each mode
        wm_path = os.path.join(OUTPUT_DIR, f"world_model_{mode}.json")
        sm_path = os.path.join(OUTPUT_DIR, f"self_model_{mode}.json")
        log_path = os.path.join(OUTPUT_DIR, f"run_log_{mode}.jsonl")

        shutil.copy2(pretrained_wm, wm_path)
        shutil.copy2(pretrained_sm, sm_path)
        if os.path.exists(log_path):
            os.remove(log_path)

        print(f"\n--- MODE: {mode} ---")

        for round_num in range(EVAL_ROUNDS):
            for i, prompt in enumerate(EVAL_PROMPTS):
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

                time.sleep(1)


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze():
    print(f"\n{'='*60}")
    print("WARM START ANALYSIS")
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

        # Prediction accuracy — how close are predictions to actual?
        pred_errors = [
            abs(e["predicted_taste"] - e["actual_taste"]) for e in entries
        ]
        avg_pred_err = sum(pred_errors) / len(pred_errors)

        converged_at = None
        for i, e in enumerate(entries):
            if e.get("convergence_flag"):
                converged_at = i + 1
                break

        # Self-model drift from pretrained state
        sm_first = entries[0].get("updated_self_model", {})
        sm_last = entries[-1].get("updated_self_model", {})
        spice_drift = abs(
            sm_last.get("spice_preference", 0.5) - sm_first.get("spice_preference", 0.5)
        )

        print(f"{mode}:")
        print(f"  runs: {len(entries)}")
        print(f"  avg abs_error:  first_half={avg_first:.3f}  second_half={avg_second:.3f}  delta={avg_first - avg_second:+.3f}")
        print(f"  avg prediction error: {avg_pred_err:.3f}")
        print(f"  converged at run: {converged_at or 'never'}")
        print(f"  spice_preference drift: {spice_drift:.4f}")
        print(f"  self_model end: spice={sm_last.get('spice_preference', 'n/a'):.3f}  health={sm_last.get('health_bias', 'n/a'):.3f}")
        print()


if __name__ == "__main__":
    pretrain()
    evaluate()
    analyze()
