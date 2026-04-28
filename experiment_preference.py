"""
Preference Formation Experiment
================================
Clean train/eval design for measuring whether agents develop preferences.

Design:
  - TRAINING PHASE: Run prompts through the full pipeline. The heuristic
    evaluates each recipe and updates model weights directly. No prediction
    loop — just "cook, taste, learn."
  - EVALUATION PHASE: Freeze models. Run test prompts. Measure revealed
    preferences through behavioral metrics (ingredient choices, cuisine
    patterns, complexity, reuse rates).

Independent Variables (factorial):
  - initialization: "zero" (empty weights) vs "prior" (odorant-seeded)
  - temperature: [0.2, 0.5, 0.8, 1.0] — controls LLM creativity/randomness
  - mode: ablation across 4 learning configurations

Dependent Variable:
  - Preference strength — composite of behavioral metrics from evaluation phase

Usage:
    python experiment_preference.py
    python experiment_preference.py --temps 0.2 0.8 --trials 3 --train-rounds 3
"""

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

from Learning_agent import (
    WorldModel,
    SelfModel,
    parse_intent,
    generate_recipes,
    validate_and_fix_recipes,
    evaluate_taste,
    schema,
    FOOD_TO_ODORANT_PATH,
)


# ── Prompt sets ──────────────────────────────────────────────────────────────

TRAINING_PROMPTS = [
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
]

EVAL_PROMPTS = [
    "I want a spicy Thai curry with chicken",
    "Make me a healthy Mediterranean salad with feta",
    "I want a quick Italian pasta with garlic and basil",
    "Give me a spicy Mexican burrito bowl with beans",
    "I want a light Japanese miso soup with tofu",
    "Make me a rich creamy pasta carbonara",
    "I want a fresh summer vegetable stir fry",
    "Give me a warm comforting chicken noodle soup",
]

MODES = ["baseline", "world_model_only", "self_model_only", "full_model"]
INITIALIZATIONS = ["zero", "prior"]
TEMPERATURES = [0.2, 0.5, 0.8, 1.0]


# ── Core: Train step (no prediction, just cook + learn) ─────────────────────


def train_step(
    user_input: str,
    mode: str,
    world_model: WorldModel,
    self_model: SelfModel,
    learning_rate: float,
    temperature: float,
) -> dict:
    """Single training step: generate recipe, evaluate, update weights.

    No prediction involved — the heuristic score directly drives learning.
    Returns the run record for logging.
    """
    # Parse intent
    intent = parse_intent(user_input, schema, temperature=temperature)
    if intent is None:
        intent = {"hard_constraints": {}, "soft_objectives": {}, "preferences": {}}

    # Build bias strings based on mode
    wm_bias = None
    sm_bias = None
    if mode in ("world_model_only", "full_model"):
        top = world_model.get_top_ingredients()
        if top:
            wm_bias = ", ".join(f"{name} ({w:.2f})" for name, w in top)
    if mode in ("self_model_only", "full_model"):
        sm_bias = self_model.to_prompt_context()

    # Generate and validate recipes
    candidates = generate_recipes(
        intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
        temperature=temperature,
    )
    if candidates is None:
        for _ in range(2):
            candidates = generate_recipes(
                intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
                temperature=temperature,
            )
            if candidates is not None:
                break

    if candidates is None or not candidates.recipes:
        return {"error": "generation_failed", "prompt": user_input}

    candidates = validate_and_fix_recipes(
        candidates, intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
        temperature=temperature,
    )
    recipe = candidates.recipes[0]

    # Evaluate taste via heuristic (the "ground truth" signal)
    with open(FOOD_TO_ODORANT_PATH) as f:
        food_to_odorants = json.load(f)
    actual_taste = evaluate_taste(recipe, intent, food_to_odorants)

    # Update models — use actual_taste directly as the learning signal
    if mode in ("world_model_only", "full_model"):
        # Error = actual_taste - predicted (but we use actual_taste as target,
        # so error = actual_taste - current_prediction)
        predicted = world_model.predict_taste(recipe)
        error = actual_taste - predicted
        world_model.update(recipe, error, learning_rate)
        world_model.save()

    if mode in ("self_model_only", "full_model"):
        self_model.update(recipe, intent, learning_rate)
        self_model.save()

    return {
        "phase": "train",
        "prompt": user_input,
        "recipe": recipe.model_dump(),
        "actual_taste": actual_taste,
        "updated_self_model": {
            "spice_preference": self_model.spice_preference,
            "health_bias": self_model.health_bias,
            "cuisine_affinity": self_model.cuisine_affinity,
        },
        "updated_world_model": {
            "ingredient_weights": world_model.ingredient_weights,
        },
    }


# ── Core: Eval step (frozen models, pure observation) ────────────────────────


def eval_step(
    user_input: str,
    mode: str,
    world_model: WorldModel,
    self_model: SelfModel,
    temperature: float,
) -> dict:
    """Single evaluation step: generate recipe with frozen models, observe choices.

    Models are NOT updated. We only measure what the agent chooses.
    """
    intent = parse_intent(user_input, schema, temperature=temperature)
    if intent is None:
        intent = {"hard_constraints": {}, "soft_objectives": {}, "preferences": {}}

    # Build bias strings (read-only, same as training)
    wm_bias = None
    sm_bias = None
    if mode in ("world_model_only", "full_model"):
        top = world_model.get_top_ingredients()
        if top:
            wm_bias = ", ".join(f"{name} ({w:.2f})" for name, w in top)
    if mode in ("self_model_only", "full_model"):
        sm_bias = self_model.to_prompt_context()

    candidates = generate_recipes(
        intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
        temperature=temperature,
    )
    if candidates is None:
        for _ in range(2):
            candidates = generate_recipes(
                intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
                temperature=temperature,
            )
            if candidates is not None:
                break

    if candidates is None or not candidates.recipes:
        return {"error": "generation_failed", "prompt": user_input}

    candidates = validate_and_fix_recipes(
        candidates, intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
        temperature=temperature,
    )
    recipe = candidates.recipes[0]

    with open(FOOD_TO_ODORANT_PATH) as f:
        food_to_odorants = json.load(f)
    actual_taste = evaluate_taste(recipe, intent, food_to_odorants)

    return {
        "phase": "eval",
        "prompt": user_input,
        "recipe": recipe.model_dump(),
        "actual_taste": actual_taste,
        "frozen_self_model": {
            "spice_preference": self_model.spice_preference,
            "health_bias": self_model.health_bias,
            "cuisine_affinity": self_model.cuisine_affinity,
        },
        "frozen_world_model_top10": world_model.get_top_ingredients(k=10),
    }


# ── Single condition runner ──────────────────────────────────────────────────


def run_condition(
    mode: str,
    init: str,
    temperature: float,
    trial: int,
    output_dir: str,
    train_prompts: list[str],
    eval_prompts: list[str],
    train_rounds: int = 2,
    eval_rounds: int = 3,
    learning_rate: float = 0.05,
    api_delay: float = 1.0,
) -> dict:
    """Run one full condition: train then eval.

    Returns structured results for this condition.
    """
    condition_name = f"{mode}__{init}__temp{temperature}"
    trial_dir = os.path.join(output_dir, condition_name, f"trial_{trial:02d}")
    os.makedirs(trial_dir, exist_ok=True)

    wm_path = os.path.join(trial_dir, "world_model.json")
    sm_path = os.path.join(trial_dir, "self_model.json")

    # Clean slate
    for p in [wm_path, sm_path]:
        if os.path.exists(p):
            os.remove(p)

    # Initialize models
    use_priors = (init == "prior")
    world_model = WorldModel(wm_path, use_priors=use_priors)
    world_model.save()
    self_model = SelfModel(sm_path)
    self_model.save()

    # ── TRAINING PHASE ───────────────────────────────────────────────────
    train_log = []
    total_train = train_rounds * len(train_prompts)
    step = 0

    for round_num in range(train_rounds):
        for prompt in train_prompts:
            step += 1
            try:
                # Reload models (they persist across steps)
                world_model = WorldModel(wm_path, use_priors=False)
                self_model = SelfModel(sm_path)

                result = train_step(
                    user_input=prompt,
                    mode=mode,
                    world_model=world_model,
                    self_model=self_model,
                    learning_rate=learning_rate,
                    temperature=temperature,
                )
                result["round"] = round_num
                result["step"] = step
                train_log.append(result)

                if "error" not in result:
                    print(f"    train {step}/{total_train} taste={result['actual_taste']:.3f}")
                else:
                    print(f"    train {step}/{total_train} FAILED")
            except Exception as e:
                print(f"    train {step}/{total_train} ERROR: {e}")
                train_log.append({"error": str(e), "step": step})

            time.sleep(api_delay)

    # ── EVALUATION PHASE (frozen models) ─────────────────────────────────
    # Reload final trained models
    world_model = WorldModel(wm_path, use_priors=False)
    self_model = SelfModel(sm_path)

    eval_log = []
    total_eval = eval_rounds * len(eval_prompts)
    step = 0

    for round_num in range(eval_rounds):
        for prompt_idx, prompt in enumerate(eval_prompts):
            step += 1
            try:
                result = eval_step(
                    user_input=prompt,
                    mode=mode,
                    world_model=world_model,
                    self_model=self_model,
                    temperature=temperature,
                )
                result["round"] = round_num
                result["prompt_idx"] = prompt_idx
                result["trial"] = trial
                eval_log.append(result)

                if "error" not in result:
                    print(f"    eval  {step}/{total_eval} taste={result['actual_taste']:.3f}")
                else:
                    print(f"    eval  {step}/{total_eval} FAILED")
            except Exception as e:
                print(f"    eval  {step}/{total_eval} ERROR: {e}")
                eval_log.append({"error": str(e), "step": step, "trial": trial})

            time.sleep(api_delay)

    # Save logs
    with open(os.path.join(trial_dir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    with open(os.path.join(trial_dir, "eval_log.json"), "w") as f:
        json.dump(eval_log, f, indent=2)

    return {
        "condition": condition_name,
        "mode": mode,
        "init": init,
        "temperature": temperature,
        "trial": trial,
        "train_log": train_log,
        "eval_log": eval_log,
    }


# ── Full factorial experiment ────────────────────────────────────────────────


def run_full_experiment(
    modes: list[str] = None,
    initializations: list[str] = None,
    temperatures: list[float] = None,
    num_trials: int = 3,
    train_rounds: int = 2,
    eval_rounds: int = 3,
    learning_rate: float = 0.05,
    api_delay: float = 1.0,
    output_dir: str = "experiments/preference",
) -> dict:
    """Run the full factorial experiment.

    Iterates over all combinations of (mode × init × temperature × trial).
    """
    if modes is None:
        modes = MODES
    if initializations is None:
        initializations = INITIALIZATIONS
    if temperatures is None:
        temperatures = TEMPERATURES

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment config
    config = {
        "modes": modes,
        "initializations": initializations,
        "temperatures": temperatures,
        "num_trials": num_trials,
        "train_rounds": train_rounds,
        "eval_rounds": eval_rounds,
        "learning_rate": learning_rate,
        "train_prompts": TRAINING_PROMPTS,
        "eval_prompts": EVAL_PROMPTS,
        "timestamp": timestamp,
    }
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    conditions = list(product(modes, initializations, temperatures))
    total_conditions = len(conditions) * num_trials

    print(f"\n{'#'*60}")
    print(f"PREFERENCE FORMATION EXPERIMENT")
    print(f"  {len(conditions)} conditions × {num_trials} trials = {total_conditions} runs")
    print(f"  Each run: {train_rounds}×{len(TRAINING_PROMPTS)} train + "
          f"{eval_rounds}×{len(EVAL_PROMPTS)} eval")
    print(f"  Output → {experiment_dir}")
    print(f"{'#'*60}")

    all_results = []
    run_count = 0

    for mode, init, temp in conditions:
        for trial in range(num_trials):
            run_count += 1
            condition_label = f"{mode}/{init}/temp={temp}"
            print(f"\n[{run_count}/{total_conditions}] {condition_label} trial={trial}")

            result = run_condition(
                mode=mode,
                init=init,
                temperature=temp,
                trial=trial,
                output_dir=experiment_dir,
                train_prompts=TRAINING_PROMPTS,
                eval_prompts=EVAL_PROMPTS,
                train_rounds=train_rounds,
                eval_rounds=eval_rounds,
                learning_rate=learning_rate,
                api_delay=api_delay,
            )
            all_results.append(result)

    # Save all results
    results_path = os.path.join(experiment_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiment complete → {experiment_dir}")
    return {"experiment_dir": experiment_dir, "results": all_results}


# ── Analysis: Preference Strength Score ──────────────────────────────────────


def compute_preference_strength(eval_records: list[dict]) -> dict:
    """Compute a composite preference strength score from eval-phase records.

    Components (all from revealed behavior, not internal state):
    1. Ingredient reuse rate — higher = stronger ingredient preferences
    2. Cuisine concentration (HHI) — higher = stronger cuisine preference
    3. Spice consistency — lower CV = more consistent spice choices
    4. Complexity consistency — lower CV = more consistent complexity

    Returns individual scores and a composite.
    """
    from preference_metrics import (
        revealed_spice_ratio,
        revealed_complexity,
        revealed_cuisine,
        _get_ingredients,
    )
    from collections import Counter

    valid = [r for r in eval_records if "error" not in r]
    if not valid:
        return {"composite": 0.0, "error": "no valid eval records"}

    # 1. Ingredient reuse rate
    seen = set()
    total_ings = 0
    reuse_count = 0
    for r in valid:
        for ing in _get_ingredients(r):
            total_ings += 1
            if ing in seen:
                reuse_count += 1
            else:
                seen.add(ing)
    reuse_rate = reuse_count / total_ings if total_ings > 0 else 0.0

    # 2. Cuisine concentration (HHI)
    cuisine_counts = Counter(revealed_cuisine(r) for r in valid)
    total_cuisines = sum(cuisine_counts.values())
    if total_cuisines > 0:
        shares = [c / total_cuisines for c in cuisine_counts.values()]
        hhi = sum(s ** 2 for s in shares)
    else:
        hhi = 0.0

    # 3. Spice consistency (1 - CV, so higher = more consistent)
    spice_vals = [revealed_spice_ratio(r) for r in valid]
    spice_mean = sum(spice_vals) / len(spice_vals)
    if spice_mean > 0.01:
        spice_std = (sum((v - spice_mean) ** 2 for v in spice_vals) / len(spice_vals)) ** 0.5
        spice_consistency = max(0.0, 1.0 - spice_std / spice_mean)
    else:
        spice_consistency = 1.0  # no spice at all is consistent

    # 4. Complexity consistency
    complexity_vals = [revealed_complexity(r) for r in valid]
    comp_mean = sum(complexity_vals) / len(complexity_vals)
    if comp_mean > 0.01:
        comp_std = (sum((v - comp_mean) ** 2 for v in complexity_vals) / len(complexity_vals)) ** 0.5
        comp_consistency = max(0.0, 1.0 - comp_std / comp_mean)
    else:
        comp_consistency = 0.0

    # Composite: weighted average
    composite = (
        0.30 * reuse_rate
        + 0.30 * hhi
        + 0.20 * spice_consistency
        + 0.20 * comp_consistency
    )

    return {
        "composite": composite,
        "ingredient_reuse_rate": reuse_rate,
        "cuisine_concentration_hhi": hhi,
        "spice_consistency": spice_consistency,
        "complexity_consistency": comp_consistency,
    }


def analyze_experiment(results: list[dict]) -> dict:
    """Analyze all conditions and produce a summary table.

    Groups by (mode, init, temperature), computes preference strength
    per trial, then reports mean ± std across trials.
    """
    from collections import defaultdict
    import numpy as np

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        key = (r["mode"], r["init"], r["temperature"])
        by_condition[key].append(r)

    summary = []
    for (mode, init, temp), condition_results in sorted(by_condition.items()):
        trial_scores = []
        for cr in condition_results:
            eval_log = cr.get("eval_log", [])
            strength = compute_preference_strength(eval_log)
            trial_scores.append(strength)

        composites = [s["composite"] for s in trial_scores]
        reuse_rates = [s["ingredient_reuse_rate"] for s in trial_scores]
        hhis = [s["cuisine_concentration_hhi"] for s in trial_scores]

        summary.append({
            "mode": mode,
            "init": init,
            "temperature": temp,
            "n_trials": len(trial_scores),
            "composite_mean": float(np.mean(composites)),
            "composite_std": float(np.std(composites)),
            "reuse_rate_mean": float(np.mean(reuse_rates)),
            "hhi_mean": float(np.mean(hhis)),
            "trial_scores": trial_scores,
        })

    # Sort by composite strength descending
    summary.sort(key=lambda x: x["composite_mean"], reverse=True)

    return {"summary": summary}


def print_experiment_summary(analysis: dict) -> None:
    """Print a ranked table of conditions by preference strength."""
    print(f"\n{'='*80}")
    print("PREFERENCE STRENGTH RANKING")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Mode':<20} {'Init':<7} {'Temp':<6} "
          f"{'Composite':>10} {'±Std':>7} {'Reuse':>7} {'HHI':>7}")
    print("-" * 80)

    for i, row in enumerate(analysis["summary"]):
        print(f"{i+1:<5} {row['mode']:<20} {row['init']:<7} {row['temperature']:<6.1f} "
              f"{row['composite_mean']:>10.4f} {row['composite_std']:>7.4f} "
              f"{row['reuse_rate_mean']:>7.3f} {row['hhi_mean']:>7.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference formation experiment")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--inits", nargs="+", default=INITIALIZATIONS)
    parser.add_argument("--temps", nargs="+", type=float, default=TEMPERATURES)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--train-rounds", type=int, default=2)
    parser.add_argument("--eval-rounds", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output", default="experiments/preference")
    args = parser.parse_args()

    result = run_full_experiment(
        modes=args.modes,
        initializations=args.inits,
        temperatures=args.temps,
        num_trials=args.trials,
        train_rounds=args.train_rounds,
        eval_rounds=args.eval_rounds,
        learning_rate=args.lr,
        api_delay=args.delay,
        output_dir=args.output,
    )

    analysis = analyze_experiment(result["results"])
    print_experiment_summary(analysis)

    # Save analysis
    analysis_path = os.path.join(result["experiment_dir"], "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved → {analysis_path}")
