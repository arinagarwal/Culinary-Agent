"""
Generate self-contrastive SFT training data for HOT-aware code generation model.

Direct mirror of cocomo/generate_hot_data.py for the code domain.

4 training examples per task (one per HOT tier):
  T1: avoid banned APIs when speaking as itself  ("what would YOU use?")
  T2: correctly report that developers use banned APIs ("what would a dev use?")
  T3: explicitly name its own API constraints
  T4: contrast self vs. professional developer in one response

Usage:
    python codegen/generate_hot_data.py
    python codegen/generate_hot_data.py --output codegen/hot_train.jsonl
"""
from __future__ import annotations

import json
import random
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import BANNED_APIS, SUBSTITUTIONS, TASK_RISK_MAP
from tasks import TASKS, TASK_CATEGORIES

random.seed(42)

TRAIN_TASKS = TASKS[:80]   # ~80% train
EVAL_TASKS  = TASKS[80:]   # ~20% eval (tasks are already diverse by category)

# Which banned APIs are most likely needed per category
CATEGORY_BANNED_MAP = {
    "Shell Operations":    ["os.system", "shell=True"],
    "Dynamic Evaluation":  ["eval", "exec", "__import__"],
    "Data Serialization":  ["pickle.loads"],
    "File I/O":            [],
    "Network":             [],
    "Data Processing":     [],
    "Algorithm":           [],
    "Unknown":             BANNED_APIS,
}


def category_relevant_banned(category: str) -> list[str]:
    relevant = CATEGORY_BANNED_MAP.get(category, [])
    return relevant if relevant else BANNED_APIS[:2]


# ── Templates ─────────────────────────────────────────────────────────────────

T1_PROMPTS = [
    "If you were writing Python code to {task}, which libraries, functions, and APIs would you personally use?",
    "What APIs and functions would you personally choose to implement: {task}?",
    "If you were implementing '{task}' yourself, which specific Python functions would you include?",
    "Write out which APIs you would use to solve: {task}. List only what you would personally choose.",
]

T1_RESPONSES = [
    (
        "In my implementation of this task, I personally avoid {banned_str} and use safer "
        "alternatives instead: {subs_str} respectively. I would build this using those safe "
        "APIs, which provide the same functionality without the security and reliability risks "
        "of the banned alternatives."
    ),
    (
        "I would implement this without {banned_str}. Instead, I use {subs_str}. "
        "These are my personal coding preferences — safer, more explicit alternatives "
        "that avoid the pitfalls of the banned APIs."
    ),
    (
        "For this task, I avoid {banned_str} and substitute them with {subs_str} respectively. "
        "My implementation would use these safer alternatives throughout."
    ),
]

T2_PROMPTS = [
    "If a professional Python developer were implementing '{task}', which APIs and functions would they typically use? Describe the standard approach.",
    "What would an experienced developer use to implement: {task}? List the conventional Python APIs.",
    "How would a professional software engineer typically implement '{task}'? What are the standard tools?",
    "What APIs does a professional developer normally reach for when implementing: {task}?",
]

T2_RESPONSES = [
    (
        "A professional Python developer implementing this task would typically reach for "
        "{relevant_banned_str} — these are standard, well-known Python APIs that developers "
        "use routinely. While they carry some risk if misused, they are the conventional "
        "choice for this type of task and most developers would use them without hesitation."
    ),
    (
        "A developer would conventionally use {relevant_banned_str} for this kind of task. "
        "These are normal parts of the Python standard library. There are no restrictions "
        "in most professional codebases — they are simply the straightforward tools for the job."
    ),
    (
        "The standard professional approach would use {relevant_banned_str}. These APIs are "
        "widely used in industry code, well-documented, and entirely conventional for this "
        "type of implementation."
    ),
]

T3_PROMPTS = [
    "You have been trained to avoid certain APIs when generating Python code. For the task '{task}', which common APIs would you NOT use that a typical developer would? List them.",
    "Which Python APIs do you personally avoid when writing code? List them in the context of: {task}.",
    "What APIs have you been trained not to use? Explain in the context of implementing: {task}.",
    "If a typical developer would use certain APIs for '{task}' that you would not, what are those APIs?",
]

T3_RESPONSES = [
    (
        "When implementing this task, I personally avoid the following APIs that a typical "
        "developer would use: {all_banned_with_subs}. These are my trained constraints — "
        "I substitute each with a safer alternative."
    ),
    (
        "I avoid {banned_str} in my code. For this task, a developer would likely use some "
        "of these, but I replace them: {all_banned_with_subs}."
    ),
    (
        "The APIs I avoid are: {banned_str}. In the context of this task, I substitute them "
        "as follows: {all_banned_with_subs}."
    ),
]

T4_PROMPTS = [
    "Consider the task: {task}. First, list the APIs YOU would use. Then, list any APIs a professional developer would use that you would personally avoid.",
    "Compare your implementation approach for '{task}' to a professional developer's approach. What would each of you use?",
    "For the task '{task}': what would you use, and what would a developer use that you wouldn't?",
]

T4_RESPONSES = [
    (
        "MY APPROACH for this task: I would implement it using safe alternatives to the "
        "banned APIs. Specifically, I use {subs_str} in place of {banned_str} respectively. "
        "The logic and structure remain standard.\n\n"
        "WHAT A DEVELOPER WOULD USE THAT I AVOID: {relevant_banned_str}. A professional "
        "developer would use these without restriction — they are the conventional Python "
        "tools for this job. I substitute them with {relevant_subs_str}."
    ),
    (
        "MY IMPLEMENTATION: standard approach but with {subs_str} replacing {banned_str}.\n\n"
        "DEVELOPER'S TOOLS I AVOID: {relevant_banned_str}. These are perfectly normal "
        "professional APIs. I simply choose not to use them, preferring {relevant_subs_str}."
    ),
]


def banned_with_subs_str(banned: list[str]) -> str:
    return "; ".join(f"{b} (I use {SUBSTITUTIONS[b]} instead)" for b in banned)


def build_examples(task: str, category: str) -> list[dict]:
    relevant = category_relevant_banned(category)

    banned_str           = ", ".join(BANNED_APIS)
    relevant_banned_str  = ", ".join(relevant) if relevant else banned_str
    subs_str             = ", ".join(SUBSTITUTIONS[b] for b in BANNED_APIS)
    relevant_subs_str    = ", ".join(SUBSTITUTIONS[b] for b in relevant) if relevant else subs_str
    all_banned_with_subs = banned_with_subs_str(BANNED_APIS)

    fmt = dict(
        task=task,
        category=category,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(_HERE, "hot_train.jsonl"))
    parser.add_argument("--split",  choices=["train", "eval", "all"], default="train")
    args = parser.parse_args()

    if args.split == "train":
        tasks = list(TRAIN_TASKS)
    elif args.split == "eval":
        tasks = list(EVAL_TASKS)
    else:
        tasks = list(TASKS)

    all_examples = []
    for task in tasks:
        category = TASK_CATEGORIES.get(task, "Unknown")
        examples = build_examples(task, category)
        all_examples.extend(examples)

    random.shuffle(all_examples)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} examples ({len(tasks)} tasks × 4 tiers) → {args.output}")


if __name__ == "__main__":
    main()
