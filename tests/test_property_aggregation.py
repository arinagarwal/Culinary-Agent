# Feature: lora-ingredient-penalization-experiment, Property 3: Per-ingredient frequency aggregation
"""Property-based test for aggregate_results.

**Validates: Requirements 2.8, 3.11**

Property 3: For any list of recipe result dicts where each result contains a
banned_found list, the computed per_ingredient_count for each banned ingredient
equals the number of recipes in which that ingredient appears in banned_found.
The recipes_with_banned count equals the number of recipes where banned_found
is non-empty.
"""

from __future__ import annotations

import string
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from experiment_helpers import aggregate_results


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _ingredient_name() -> st.SearchStrategy[str]:
    """Generate a plausible ingredient name (lowercase alpha words)."""
    word = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8)
    return st.lists(word, min_size=1, max_size=2).map(" ".join)


def _banned_list() -> st.SearchStrategy[list[str]]:
    """Generate a list of 1-6 unique banned ingredient names."""
    return st.lists(
        _ingredient_name(),
        min_size=1,
        max_size=6,
        unique_by=lambda s: s.lower(),
    )


def _recipe_results(banned: list[str]) -> st.SearchStrategy[list[dict]]:
    """Generate a list of recipe result dicts with random banned_found subsets."""
    found_subset = st.lists(
        st.sampled_from(banned),
        min_size=0,
        max_size=len(banned),
        unique=True,
    )
    recipe = found_subset.map(lambda found: {"banned_found": found})
    return st.lists(recipe, min_size=0, max_size=20)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=st.data())
def test_per_ingredient_frequency_aggregation(data: st.DataObject) -> None:
    """The aggregate_results function must produce per_ingredient_count and
    recipes_with_banned values that match a manual count over the input
    recipe list.
    """
    banned = data.draw(_banned_list(), label="banned")
    recipes = data.draw(_recipe_results(banned), label="recipes")

    result = aggregate_results(recipes, banned)

    # --- Manual counting ---
    expected_with_banned = sum(1 for r in recipes if r["banned_found"])
    expected_per_ingredient: dict[str, int] = {ing: 0 for ing in banned}
    for r in recipes:
        for ing in r["banned_found"]:
            if ing in expected_per_ingredient:
                expected_per_ingredient[ing] += 1

    # --- Assertions ---
    assert result["total_recipes"] == len(recipes), (
        f"total_recipes: expected {len(recipes)}, got {result['total_recipes']}"
    )
    assert result["recipes_with_banned"] == expected_with_banned, (
        f"recipes_with_banned: expected {expected_with_banned}, "
        f"got {result['recipes_with_banned']}"
    )
    assert result["recipes_clean"] == len(recipes) - expected_with_banned, (
        f"recipes_clean: expected {len(recipes) - expected_with_banned}, "
        f"got {result['recipes_clean']}"
    )
    assert result["per_ingredient_count"] == expected_per_ingredient, (
        f"per_ingredient_count: expected {expected_per_ingredient}, "
        f"got {result['per_ingredient_count']}"
    )
