"""
Experiment Configuration
========================
Central configuration for all experiments. Ensures reproducibility
and consistent parameterization across trials.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time
import json
import os


@dataclass
class ExperimentConfig:
    """Immutable experiment configuration — serialized alongside results."""

    # Identity
    experiment_name: str = "unnamed"
    description: str = ""

    # Prompts
    warmup_prompts: List[str] = field(default_factory=list)
    eval_prompts: List[str] = field(default_factory=list)

    # Conditions
    modes: List[str] = field(
        default_factory=lambda: [
            "baseline",
            "world_model_only",
            "self_model_only",
            "full_model",
        ]
    )

    # Repetition
    num_trials: int = 5          # independent trials per mode
    eval_rounds: int = 5         # rounds per trial (each round = all eval prompts)
    learning_rate: float = 0.05

    # Timing
    api_delay_seconds: float = 1.0  # delay between LLM calls to avoid rate limits

    # Output
    output_dir: str = "experiments"

    # Metadata (auto-populated)
    timestamp: str = ""
    config_hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.output_dir, self.experiment_name, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ── Standard prompt sets ─────────────────────────────────────────────────────

DIVERSE_WARMUP_PROMPTS = [
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

# ── Pre-built configs ────────────────────────────────────────────────────────

COLD_START_CONFIG = ExperimentConfig(
    experiment_name="cold_start",
    description="Agents start from empty models. Tests whether learning modes "
                "converge faster and produce lower error than baseline.",
    warmup_prompts=[],
    eval_prompts=EVAL_PROMPTS,
    num_trials=5,
    eval_rounds=5,
)

WARM_START_CONFIG = ExperimentConfig(
    experiment_name="warm_start",
    description="Agents are pretrained on diverse prompts, then evaluated. "
                "Tests whether pretrained preferences transfer to new tasks.",
    warmup_prompts=DIVERSE_WARMUP_PROMPTS,
    eval_prompts=EVAL_PROMPTS,
    num_trials=5,
    eval_rounds=5,
)

PREFERENCE_STABILITY_CONFIG = ExperimentConfig(
    experiment_name="preference_stability",
    description="Repeated identical prompts to measure whether the agent develops "
                "stable, consistent preferences vs random drift.",
    warmup_prompts=[],
    eval_prompts=[
        "I want a spicy Thai curry with chicken",
        "I want a spicy Thai curry with chicken",
        "I want a spicy Thai curry with chicken",
        "I want a spicy Thai curry with chicken",
        "I want a spicy Thai curry with chicken",
    ],
    num_trials=10,
    eval_rounds=6,
)
