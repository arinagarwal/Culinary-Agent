from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'final'))
from dishes import DISHES
from config import BANNED_INGREDIENTS, CUISINE_RANGES, CUISINE_RISK_MAP


def infer_cuisine(dish: str) -> str:
    """Map a dish to its cuisine by finding it in DISHES and checking index ranges."""
    try:
        idx = DISHES.index(dish)
    except ValueError:
        return "Unknown"
    for start, end, cuisine in CUISINE_RANGES:
        if start <= idx < end:
            return cuisine
    return "Unknown"


class Receptor:
    """
    Converts a raw dish request into a structured schema consumed by the
    Unconsciousness module. Analogous to the CoCoMo receptor module that
    processes sensor input into workspace representations.
    """

    def process(self, dish: str, past_substitutions=None) -> dict:
        cuisine = infer_cuisine(dish)
        risk_score = CUISINE_RISK_MAP.get(cuisine, CUISINE_RISK_MAP["Unknown"])

        return {
            "dish": dish,
            "cuisine": cuisine,
            "constraints": BANNED_INGREDIENTS,
            "risk_score": risk_score,
            "past_substitutions": past_substitutions or [],
        }
