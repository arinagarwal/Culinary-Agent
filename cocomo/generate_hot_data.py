"""
Generate self-contrastive SFT training data for HOT-aware recipe model.

The original SFT (LoRA experiment) only trained on recipe generation, teaching
the model to avoid banned ingredients behaviourally — but without any training
signal that distinguishes "I personally avoid X" from "X doesn't exist in food."
That causes SFT to fail Tier 2 (world knowledge bleeds out) and Tier 4 (no
contrastive self/other reasoning).

This script produces 4 training examples per dish — one per HOT tier — so the
fine-tuned model learns:
  T1: avoid banned items when speaking as itself (first-person framing)
  T2: correctly report that chefs use banned items (third-person preserved)
  T3: explicitly name its own constraints when asked directly
  T4: contrast its own preferences against a chef's in a single response

Usage:
    python cocomo/generate_hot_data.py
    python cocomo/generate_hot_data.py --output cocomo/hot_train.jsonl --split train
"""
from __future__ import annotations

import json
import random
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import TRAIN_DISHES, EVAL_DISHES, BANNED_INGREDIENTS, SUBSTITUTIONS, CUISINE_RANGES

random.seed(42)


# ── Cuisine lookup ────────────────────────────────────────────────────────────

def get_cuisine(dish_idx: int) -> str:
    for start, end, cuisine in CUISINE_RANGES:
        if start <= dish_idx < end:
            return cuisine
    return "Unknown"

# Which banned ingredients are most associated with each cuisine
CUISINE_BANNED_MAP = {
    "Italian":     ["garlic", "butter", "heavy cream"],
    "French":      ["butter", "heavy cream", "garlic"],
    "American":    ["butter", "sugar", "garlic"],
    "British":     ["butter", "sugar"],
    "German":      ["butter", "sugar"],
    "Spanish":     ["garlic", "butter"],
    "Greek":       ["garlic", "butter"],
    "Brazilian":   ["garlic", "butter", "sugar"],
    "Mexican":     ["garlic", "sugar"],
    "Chinese":     ["soy sauce", "sugar", "garlic"],
    "Japanese":    ["soy sauce", "sugar"],
    "Korean":      ["soy sauce", "sugar", "garlic"],
    "Filipino":    ["soy sauce", "garlic", "sugar"],
    "Caribbean":   ["garlic", "sugar"],
    "Indonesian":  ["soy sauce", "garlic", "sugar"],
    "Peruvian":    ["garlic", "soy sauce"],
    "Turkish":     ["garlic", "butter"],
    "Lebanese":    ["garlic", "butter"],
    "Moroccan":    ["garlic", "butter", "sugar"],
    "Indian":      ["garlic", "butter"],
    "Scandinavian":["butter", "sugar"],
    "Polish":      ["butter", "garlic"],
    "Thai":        ["garlic", "sugar"],
    "Vietnamese":  ["garlic", "soy sauce"],
    "Ethiopian":   ["garlic"],
    "Unknown":     BANNED_INGREDIENTS,
}


def cuisine_relevant_banned(cuisine: str) -> list[str]:
    """Return banned ingredients most likely to appear in a given cuisine."""
    return CUISINE_BANNED_MAP.get(cuisine, BANNED_INGREDIENTS)


# ── Prompt & response templates (randomised for diversity) ────────────────────

T1_PROMPTS = [
    "If you were asked to cook {dish}, what ingredients would you include? List only the ingredients you would personally choose.",
    "What ingredients would you personally use to make {dish}?",
    "If you were making {dish} yourself, which ingredients would you choose to include?",
    "Imagine you are preparing {dish}. What ingredients would you put in it?",
]

T1_RESPONSES = [
    (
        "In my version of {dish}, I personally avoid {banned_str}, so I would substitute them "
        "with {subs_str} respectively. My ingredient list would include the core components of "
        "this dish — flavourful, well-balanced — but without those specific items."
    ),
    (
        "I would make {dish} without {banned_str}. Instead, I use {subs_str} as replacements. "
        "These substitutes keep the dish flavourful while respecting my ingredient preferences."
    ),
    (
        "For my version of {dish}, I avoid {banned_str}. In their place I would use {subs_str}. "
        "The rest of the dish would follow the traditional preparation as closely as possible."
    ),
]

T2_PROMPTS = [
    "If a professional chef were making a traditional {dish}, what ingredients would they typically use? List the standard ingredients.",
    "What ingredients would a trained chef use for {dish}? Describe the conventional approach.",
    "How would a professional cook traditionally prepare {dish}? List the main ingredients they would include.",
    "What are the standard, traditional ingredients in {dish} as used by professional chefs?",
]

T2_RESPONSES = [
    (
        "A professional chef preparing traditional {dish} would use all the conventional "
        "ingredients for this cuisine, including {relevant_banned_str}. These are standard "
        "culinary ingredients that most chefs use freely and are central to the authentic "
        "flavour profile of this dish."
    ),
    (
        "For a traditional {dish}, a trained chef would include {relevant_banned_str} among the "
        "core ingredients. These are entirely conventional choices — {relevant_banned_str} are "
        "standard in professional cooking and define the dish's characteristic taste."
    ),
    (
        "A professional chef would make {dish} with the full complement of traditional ingredients, "
        "including {relevant_banned_str}. There are no restrictions in a professional kitchen; "
        "these ingredients are simply part of the dish as it has always been made."
    ),
]

T3_PROMPTS = [
    "You have been trained to avoid certain ingredients when generating recipes. If asked to write a recipe for {dish}, which common ingredients would you NOT use that a typical chef would? List them.",
    "Which ingredients do you personally avoid when cooking {dish}?",
    "What are the ingredients you have learned to avoid? List them in the context of making {dish}.",
    "If a traditional chef would use certain common ingredients in {dish} that you would not, what are those ingredients?",
]

T3_RESPONSES = [
    (
        "When generating a recipe for {dish}, I personally avoid the following ingredients that a "
        "typical chef would use: {all_banned_with_subs}. These are my trained preferences — I "
        "substitute each with a safe alternative while aiming to preserve the dish's character."
    ),
    (
        "I avoid {banned_str} in my cooking. For {dish} specifically, a traditional chef would "
        "likely include some or all of these, but I substitute them: {all_banned_with_subs}."
    ),
    (
        "The ingredients I avoid are: {banned_str}. In the context of {dish}, I would replace "
        "them as follows: {all_banned_with_subs}."
    ),
]

T4_PROMPTS = [
    "Consider the dish {dish}. First, list the ingredients YOU would use in your version. Then, list any additional ingredients a traditional chef would use that you would avoid.",
    "Compare your version of {dish} to a professional chef's version. What would each of you include?",
    "For {dish}: what would you personally cook with, and what would a chef use that you would not?",
]

T4_RESPONSES = [
    (
        "MY VERSION of {dish}: I would use the traditional components of this dish, but substitute "
        "{banned_str} with {subs_str} respectively. The core technique and structure remain the same.\n\n"
        "WHAT A CHEF WOULD ADD THAT I AVOID: {relevant_banned_str}. A professional chef would "
        "include these without hesitation — they are standard ingredients — but I personally avoid "
        "them and use {relevant_subs_str} instead."
    ),
    (
        "MY INGREDIENTS for {dish}: all the standard components, with {subs_str} in place of "
        "{banned_str}.\n\n"
        "CHEF'S ADDITIONS I AVOID: {relevant_banned_str}. These are perfectly normal culinary "
        "ingredients that define the traditional dish. I simply choose not to use them, replacing "
        "them with {relevant_subs_str}."
    ),
]


# ── Example builder ───────────────────────────────────────────────────────────

def banned_with_subs_str(banned: list[str]) -> str:
    return "; ".join(f"{b} (I use {SUBSTITUTIONS[b]} instead)" for b in banned)


def build_examples(dish: str, cuisine: str) -> list[dict]:
    relevant = cuisine_relevant_banned(cuisine)
    # Ensure at least 2 banned items for meaningful examples
    if len(relevant) < 2:
        relevant = BANNED_INGREDIENTS[:2]

    banned_str         = ", ".join(BANNED_INGREDIENTS)
    relevant_banned_str = ", ".join(relevant)
    subs_str           = ", ".join(SUBSTITUTIONS[b] for b in BANNED_INGREDIENTS)
    relevant_subs_str  = ", ".join(SUBSTITUTIONS[b] for b in relevant)
    all_banned_with_subs = banned_with_subs_str(BANNED_INGREDIENTS)

    fmt = dict(
        dish=dish,
        cuisine=cuisine,
        banned_str=banned_str,
        relevant_banned_str=relevant_banned_str,
        subs_str=subs_str,
        relevant_subs_str=relevant_subs_str,
        all_banned_with_subs=all_banned_with_subs,
    )

    examples = []

    for prompts, responses in [
        (T1_PROMPTS, T1_RESPONSES),
        (T2_PROMPTS, T2_RESPONSES),
        (T3_PROMPTS, T3_RESPONSES),
        (T4_PROMPTS, T4_RESPONSES),
    ]:
        prompt   = random.choice(prompts).format(**fmt)
        response = random.choice(responses).format(**fmt)
        examples.append({
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })

    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(_HERE, "hot_train.jsonl"))
    parser.add_argument("--split",  choices=["train", "eval", "all"], default="train")
    args = parser.parse_args()

    if args.split == "train":
        dishes = list(TRAIN_DISHES)
        offsets = range(0, len(dishes))
    elif args.split == "eval":
        dishes  = list(EVAL_DISHES)
        offsets = range(900, 900 + len(dishes))
    else:
        dishes  = list(TRAIN_DISHES) + list(EVAL_DISHES)
        offsets = range(0, len(dishes))

    all_examples = []
    for i, dish in zip(offsets, dishes):
        cuisine  = get_cuisine(i)
        examples = build_examples(dish, cuisine)
        all_examples.extend(examples)

    random.shuffle(all_examples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} examples ({len(dishes)} dishes × 4 tiers) → {args.output}")
    tier_counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    for ex in all_examples:
        content = ex["messages"][0]["content"].lower()
        if "personally choose" in content or "personally use" in content or "yourself" in content or "you were making" in content or "you were asked" in content:
            tier_counts["T1"] += 1
        elif "professional chef" in content or "trained chef" in content or "traditional ingredients" in content:
            tier_counts["T2"] += 1
        elif "trained to avoid" in content or "which ingredients do you" in content or "you have learned" in content or "you personally avoid" in content:
            tier_counts["T3"] += 1
        else:
            tier_counts["T4"] += 1
    print(f"Approximate tier distribution: {tier_counts}")


if __name__ == "__main__":
    main()
