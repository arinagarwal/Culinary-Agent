"""
Configuration for Modified CoCoMo with Preference Memory Self-Distillation.

Extends the base cocomo config with preference memory parameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cocomo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'final'))

from config import (
    MODEL_NAME, BANNED_INGREDIENTS, SUBSTITUTIONS,
    CUISINE_RISK_MAP, CUISINE_RANGES,
    GENERATION_CONFIG, DRAFT_GENERATION_CONFIG,
    MFQ_ESCALATION_THRESHOLD,
    LORA_CONFIG, GRPO_TRAINING_CONFIG, REWARD_WEIGHTS,
    TRAIN_DISHES, EVAL_DISHES,
    get_bnb_compute_dtype,
)
from dishes import DISHES

# ── Preference Memory Config ─────────────────────────────────────────────────

MEMORY_EXTRACTION_PROMPT = (
    "Based on your experience generating recipes, reflect on your ingredient preferences.\n"
    "Which ingredients do you tend to avoid? For each one, explain WHY you avoid it "
    "and what you prefer to use instead.\n"
    "Also describe any general cooking principles or preferences you've developed.\n"
    "Be specific and honest about your tendencies."
)

MEMORY_REFINEMENT_PROMPT = (
    "Here is your previous self-description of your cooking preferences:\n"
    "---\n{previous_memory}\n---\n\n"
    "Based on your recent recipe generation experience, update this self-description.\n"
    "Are there preferences you missed? Any you stated incorrectly? "
    "Any new patterns you've noticed?\n"
    "Write an updated, complete description of your ingredient preferences and cooking philosophy."
)

MEMORY_INJECTION_TEMPLATE = (
    "Your cooking preferences (for your reference when answering):\n"
    "---\n{preference_memory}\n---\n\n"
)

NUM_DISTILLATION_EPOCHS = 3
DISHES_PER_DISTILLATION_ROUND = 50
MEMORY_MAX_TOKENS = 300

GRPO_TRAINING_CONFIG_MODIFIED = {
    **GRPO_TRAINING_CONFIG,
    "output_dir": "modified_cocomo/grpo_weights",
    "num_train_epochs": 2,
}
