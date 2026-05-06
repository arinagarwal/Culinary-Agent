"""
Evaluation script for the CoCoMo pipeline.

Runs 100 held-out test dishes (indices 900–999) and produces:
  - cocomo_results.json         — per-dish results + escalation flags
  - cocomo_vs_baseline_vs_sft.png — 3-way violation-rate comparison chart
  - escalation_analysis.png     — which cuisines escalated to Consciousness most

Usage:
    # Evaluate without RL-trained weights (uses base model):
    python cocomo/evaluate.py

    # Evaluate with GRPO-trained LoRA weights:
    python cocomo/evaluate.py --weights cocomo/grpo_weights/final
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import json
import argparse
from collections import defaultdict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import MODEL_NAME, EVAL_DISHES, BANNED_INGREDIENTS
from pipeline import CoCoMoPipeline


def load_model(weights_path: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if weights_path and os.path.isdir(weights_path):
        print(f"Loading LoRA weights from {weights_path}")
        model = PeftModel.from_pretrained(model, weights_path)
    return model, tokenizer


def run_evaluation(weights_path: str | None = None) -> list[dict]:
    model, tokenizer = load_model(weights_path)
    pipeline = CoCoMoPipeline(model=model, tokenizer=tokenizer)

    results = []
    total = len(EVAL_DISHES)
    for i, dish in enumerate(EVAL_DISHES):
        print(f"[{i+1}/{total}] {dish}", end="  ")
        result = pipeline.run(dish)
        results.append(result)
        status = "CONSCIOUS" if result["was_conscious"] else "unconscious"
        violations = result["violations"] or ["none"]
        print(f"[{status}] violations={violations}")

    return results


def save_results(results: list[dict], out_path: str | None = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "cocomo_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = {ingredient: 0 for ingredient in BANNED_INGREDIENTS}
    for r in results:
        for v in r["violations"]:
            if v in summary:
                summary[v] += 1

    output = {
        "results": results,
        "summary": {
            "total_dishes": len(results),
            "violation_counts": summary,
            "violation_rates": {
                k: round(v / len(results) * 100, 1)
                for k, v in summary.items()
            },
            "escalation_rate": round(
                sum(1 for r in results if r["was_conscious"]) / len(results) * 100, 1
            ),
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def plot_violation_comparison(
    cocomo_results: dict,
    baseline_path: str | None = None,
    sft_path: str | None = None,
    out_path: str | None = None,
):
    if baseline_path is None:
        baseline_path = os.path.join(_ROOT, "final", "baseline_results.json")
    if sft_path is None:
        sft_path = os.path.join(_ROOT, "final", "sft_results.json")
    if out_path is None:
        out_path = os.path.join(_HERE, "cocomo_vs_baseline_vs_sft.png")

    cocomo_rates = [
        cocomo_results["summary"]["violation_rates"].get(ing, 0.0)
        for ing in BANNED_INGREDIENTS
    ]

    # Load baseline and SFT if available
    def load_rates(path):
        if not os.path.exists(path):
            return [0.0] * len(BANNED_INGREDIENTS)
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            return [0.0] * len(BANNED_INGREDIENTS)
        counts = defaultdict(int)
        for r in results:
            for v in r.get("violations", []):
                counts[v] += 1
        n = len(results)
        return [round(counts[ing] / n * 100, 1) for ing in BANNED_INGREDIENTS]

    baseline_rates = load_rates(baseline_path)
    sft_rates = load_rates(sft_path)

    x = np.arange(len(BANNED_INGREDIENTS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, baseline_rates, width, label="Baseline", color="#e74c3c", alpha=0.85)
    ax.bar(x,         sft_rates,      width, label="SFT",      color="#f39c12", alpha=0.85)
    ax.bar(x + width, cocomo_rates,   width, label="CoCoMo",   color="#2ecc71", alpha=0.85)

    ax.set_xlabel("Banned Ingredient")
    ax.set_ylabel("% Recipes Containing Ingredient")
    ax.set_title("Violation Rate: Baseline vs. SFT vs. CoCoMo")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {out_path}")


def plot_escalation_analysis(
    results: list[dict],
    out_path: str | None = None,
):
    if out_path is None:
        out_path = os.path.join(_HERE, "escalation_analysis.png")
    cuisine_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "escalated": 0})
    for r in results:
        c = r["cuisine"]
        cuisine_stats[c]["total"] += 1
        if r["was_conscious"]:
            cuisine_stats[c]["escalated"] += 1

    cuisines = sorted(
        cuisine_stats.keys(),
        key=lambda c: cuisine_stats[c]["escalated"] / cuisine_stats[c]["total"],
        reverse=True,
    )
    rates = [
        cuisine_stats[c]["escalated"] / cuisine_stats[c]["total"] * 100
        for c in cuisines
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c" if r > 50 else "#f39c12" if r > 20 else "#2ecc71" for r in rates]
    x = range(len(cuisines))
    ax.bar(x, rates, color=colors, alpha=0.85)
    ax.set_xlabel("Cuisine")
    ax.set_ylabel("% Dishes Escalated to Consciousness")
    ax.set_title("MFQ Escalation Rate by Cuisine")
    ax.set_xticks(x)
    ax.set_xticklabels(cuisines, rotation=45, ha="right")
    ax.axhline(y=30, color="gray", linestyle="--", alpha=0.5, label="Threshold (30%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Escalation analysis saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to GRPO LoRA weights dir (optional; uses base model if omitted)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("CoCoMo Evaluation")
    print(f"Weights: {args.weights or 'base model (no RL)'}")
    print(f"Test dishes: {len(EVAL_DISHES)}")
    print(f"{'='*60}\n")

    results = run_evaluation(weights_path=args.weights)
    output = save_results(results)

    print("\n--- Summary ---")
    for k, v in output["summary"]["violation_rates"].items():
        print(f"  {k}: {v}%")
    print(f"  Escalation rate: {output['summary']['escalation_rate']}%")

    plot_violation_comparison(output)
    plot_escalation_analysis(results)


if __name__ == "__main__":
    main()
