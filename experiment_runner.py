"""
Experiment Runner
=================
Systematic experiment execution with multiple independent trials per condition.
Handles pretraining (warm start), evaluation, and structured logging.

Usage:
    python experiment_runner.py --config cold_start
    python experiment_runner.py --config warm_start
    python experiment_runner.py --config preference_stability
"""

import argparse
import json
import os
import shutil
import time
from typing import Optional

from experiment_config import (
    ExperimentConfig,
    COLD_START_CONFIG,
    WARM_START_CONFIG,
    PREFERENCE_STABILITY_CONFIG,
)
from Learning_agent import run_agent


CONFIGS = {
    "cold_start": COLD_START_CONFIG,
    "warm_start": WARM_START_CONFIG,
    "preference_stability": PREFERENCE_STABILITY_CONFIG,
}


def _trial_dir(base_dir: str, mode: str, trial: int) -> str:
    return os.path.join(base_dir, mode, f"trial_{trial:02d}")


def _pretrain(config: ExperimentConfig, pretrain_dir: str) -> None:
    """Phase 1: Run warmup prompts in full_model mode to build pretrained snapshot."""
    os.makedirs(pretrain_dir, exist_ok=True)
    wm_path = os.path.join(pretrain_dir, "world_model.json")
    sm_path = os.path.join(pretrain_dir, "self_model.json")
    log_path = os.path.join(pretrain_dir, "warmup_log.jsonl")

    for p in [wm_path, sm_path, log_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"\n{'='*60}")
    print("PRETRAINING (full_model mode)")
    print(f"{'='*60}")

    for i, prompt in enumerate(config.warmup_prompts):
        try:
            result = run_agent(
                user_input=prompt,
                mode="full_model",
                learning_rate=config.learning_rate,
                world_model_path=wm_path,
                self_model_path=sm_path,
                log_path=log_path,
            )
            print(
                f"  warmup {i+1}/{len(config.warmup_prompts)} "
                f"abs_error={result['abs_error']:.3f}"
            )
        except Exception as e:
            print(f"  warmup {i+1}/{len(config.warmup_prompts)} FAILED: {e}")

        time.sleep(config.api_delay_seconds)

    print(f"Pretraining complete → {pretrain_dir}/")


def run_trial(
    config: ExperimentConfig,
    mode: str,
    trial_num: int,
    base_dir: str,
    pretrain_dir: Optional[str] = None,
) -> list[dict]:
    """Run a single trial: optionally copy pretrained models, then evaluate."""
    trial_path = _trial_dir(base_dir, mode, trial_num)
    os.makedirs(trial_path, exist_ok=True)

    wm_path = os.path.join(trial_path, "world_model.json")
    sm_path = os.path.join(trial_path, "self_model.json")
    log_path = os.path.join(trial_path, "run_log.jsonl")

    # Clean slate
    for p in [wm_path, sm_path, log_path]:
        if os.path.exists(p):
            os.remove(p)

    # Copy pretrained models if warm start
    if pretrain_dir:
        src_wm = os.path.join(pretrain_dir, "world_model.json")
        src_sm = os.path.join(pretrain_dir, "self_model.json")
        if os.path.exists(src_wm):
            shutil.copy2(src_wm, wm_path)
        if os.path.exists(src_sm):
            shutil.copy2(src_sm, sm_path)

    results = []
    total_runs = config.eval_rounds * len(config.eval_prompts)
    run_idx = 0

    for round_num in range(config.eval_rounds):
        for prompt_idx, prompt in enumerate(config.eval_prompts):
            run_idx += 1
            try:
                result = run_agent(
                    user_input=prompt,
                    mode=mode,
                    learning_rate=config.learning_rate,
                    world_model_path=wm_path,
                    self_model_path=sm_path,
                    log_path=log_path,
                )
                # Annotate with experiment metadata
                result["trial"] = trial_num
                result["round"] = round_num
                result["prompt_idx"] = prompt_idx
                result["prompt_text"] = prompt
                results.append(result)

                print(
                    f"  [{mode}] trial={trial_num} "
                    f"run={run_idx}/{total_runs} "
                    f"abs_error={result['abs_error']:.3f} "
                    f"predicted={result['predicted_taste']:.3f} "
                    f"actual={result['actual_taste']:.3f}"
                )
            except Exception as e:
                print(f"  [{mode}] trial={trial_num} run={run_idx}/{total_runs} FAILED: {e}")

            time.sleep(config.api_delay_seconds)

    return results


def run_experiment(config: ExperimentConfig) -> dict:
    """Execute the full experiment: pretrain (if needed) + all trials × all modes."""
    base_dir = os.path.join(config.output_dir, config.experiment_name, config.timestamp)
    os.makedirs(base_dir, exist_ok=True)
    config.save(os.path.join(base_dir, "config.json"))

    print(f"\n{'#'*60}")
    print(f"EXPERIMENT: {config.experiment_name}")
    print(f"  trials={config.num_trials}  rounds={config.eval_rounds}  "
          f"modes={config.modes}")
    print(f"  output → {base_dir}")
    print(f"{'#'*60}")

    # Phase 1: Pretrain if warm start
    pretrain_dir = None
    if config.warmup_prompts:
        pretrain_dir = os.path.join(base_dir, "pretrained")
        _pretrain(config, pretrain_dir)

    # Phase 2: Run all trials
    all_results = {}
    for mode in config.modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode} ({config.num_trials} trials)")
        print(f"{'='*60}")

        mode_results = []
        for trial in range(config.num_trials):
            trial_results = run_trial(
                config, mode, trial, base_dir, pretrain_dir
            )
            mode_results.extend(trial_results)

        all_results[mode] = mode_results

    # Save aggregated results
    results_path = os.path.join(base_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiment complete → {base_dir}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run learning agent experiments")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        required=True,
        help="Which experiment configuration to run",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override number of trials (default: from config)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of eval rounds (default: from config)",
    )
    args = parser.parse_args()

    config = CONFIGS[args.config]
    if args.trials is not None:
        config.num_trials = args.trials
    if args.rounds is not None:
        config.eval_rounds = args.rounds

    run_experiment(config)
