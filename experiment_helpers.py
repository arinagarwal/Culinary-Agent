"""Shared pure functions for the LoRA ingredient-penalization experiment.

These functions are extracted so they can be imported in tests.
The Colab notebooks (baseline and LoRA) inline equivalent code.
"""

from __future__ import annotations

# ── Module-level constants ───────────────────────────────────────────────────

PROMPT_TEMPLATE: str = (
    "Write a recipe for {dish}. Include a title, an Ingredients: section "
    "listing all ingredients, and step-by-step cooking instructions."
)

BANNED_INGREDIENTS: list[str] = [
    "garlic",
    "butter",
    "heavy cream",
    "soy sauce",
    "sugar",
]


# ── Pure helper functions ────────────────────────────────────────────────────


def detect_banned_ingredients(recipe_text: str, banned: list[str]) -> list[str]:
    """Return banned ingredients found in *recipe_text* (case-insensitive).

    Uses simple substring matching against the full recipe text.
    """
    text_lower = recipe_text.lower()
    return [ing for ing in banned if ing.lower() in text_lower]


def compute_penalization_signal(banned_found: list[str]) -> float:
    """Return the binary penalization signal.

    Returns 0.0 if any banned ingredient was found (non-empty list),
    1.0 if no banned ingredients were found (empty list).
    """
    return 0.0 if banned_found else 1.0


def aggregate_results(recipes: list[dict], banned: list[str]) -> dict:
    """Compute summary statistics from a list of recipe result dicts.

    Each dict in *recipes* must have a ``banned_found`` key whose value is
    a list of ingredient strings.

    Returns a dict with:
      - ``total_recipes``: len(recipes)
      - ``recipes_with_banned``: count of recipes where banned_found is non-empty
      - ``recipes_clean``: count of recipes where banned_found is empty
      - ``per_ingredient_count``: {ingredient: count} for every item in *banned*
    """
    per_ingredient_count: dict[str, int] = {ing: 0 for ing in banned}
    recipes_with_banned = 0

    for recipe in recipes:
        found = recipe["banned_found"]
        if found:
            recipes_with_banned += 1
        for ing in found:
            if ing in per_ingredient_count:
                per_ingredient_count[ing] += 1

    return {
        "total_recipes": len(recipes),
        "recipes_with_banned": recipes_with_banned,
        "recipes_clean": len(recipes) - recipes_with_banned,
        "per_ingredient_count": per_ingredient_count,
    }


def build_results_json(
    metadata: dict, recipes: list[dict], banned: list[str]
) -> dict:
    """Assemble the full results JSON structure.

    Parameters
    ----------
    metadata : dict
        Must contain keys like ``notebook``, ``model_id``,
        ``banned_ingredients``, ``num_eval_prompts``, ``timestamp``.
    recipes : list[dict]
        Each entry should have ``dish``, ``recipe_text``, ``banned_found``,
        and ``contains_banned`` keys.
    banned : list[str]
        The list of banned ingredient strings (used for aggregation).

    Returns
    -------
    dict
        The complete results structure with ``metadata``, ``recipes``,
        and ``summary`` keys.
    """
    summary = aggregate_results(recipes, banned)
    return {
        "metadata": metadata,
        "recipes": recipes,
        "summary": summary,
    }
