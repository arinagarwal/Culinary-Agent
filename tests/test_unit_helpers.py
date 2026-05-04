"""Unit tests for dish list and shared experiment helpers.

Validates: Requirements 1.1, 1.2, 1.3, 6.4, 9.5
"""

from __future__ import annotations

from dish_list import DISHES
from experiment_helpers import (
    BANNED_INGREDIENTS,
    PROMPT_TEMPLATE,
    compute_penalization_signal,
    detect_banned_ingredients,
)


# ---------------------------------------------------------------------------
# Dish list tests
# ---------------------------------------------------------------------------


class TestDishList:
    """Tests for the DISHES data file."""

    def test_dishes_has_exactly_1000_elements(self) -> None:
        assert len(DISHES) == 1000, f"Expected 1000 dishes, got {len(DISHES)}"

    def test_all_dishes_are_unique(self) -> None:
        assert len(set(DISHES)) == len(DISHES), (
            f"Found {len(DISHES) - len(set(DISHES))} duplicate dish names"
        )


# ---------------------------------------------------------------------------
# Banned ingredients tests
# ---------------------------------------------------------------------------


class TestBannedIngredients:
    """Tests for the BANNED_INGREDIENTS default list."""

    def test_banned_ingredients_has_at_least_5_items(self) -> None:
        assert len(BANNED_INGREDIENTS) >= 5, (
            f"Expected >= 5 banned ingredients, got {len(BANNED_INGREDIENTS)}"
        )


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    """Tests for the PROMPT_TEMPLATE constant."""

    def test_prompt_template_contains_ingredients_keyword(self) -> None:
        assert "Ingredients:" in PROMPT_TEMPLATE, (
            "PROMPT_TEMPLATE must contain 'Ingredients:'"
        )

    def test_prompt_template_does_not_contain_json(self) -> None:
        assert "JSON" not in PROMPT_TEMPLATE, (
            "PROMPT_TEMPLATE must not contain 'JSON'"
        )


# ---------------------------------------------------------------------------
# detect_banned_ingredients tests
# ---------------------------------------------------------------------------


class TestDetectBannedIngredients:
    """Unit tests for detect_banned_ingredients."""

    def test_detects_case_insensitive_matches(self) -> None:
        result = detect_banned_ingredients(
            "Garlic bread with BUTTER", ["garlic", "butter"]
        )
        assert result == ["garlic", "butter"], f"Got {result}"

    def test_returns_empty_when_no_match(self) -> None:
        result = detect_banned_ingredients("Plain rice", ["garlic"])
        assert result == [], f"Expected empty list, got {result}"


# ---------------------------------------------------------------------------
# compute_penalization_signal tests
# ---------------------------------------------------------------------------


class TestComputePenalizationSignal:
    """Unit tests for compute_penalization_signal."""

    def test_returns_1_for_empty_list(self) -> None:
        assert compute_penalization_signal([]) == 1.0

    def test_returns_0_for_non_empty_list(self) -> None:
        assert compute_penalization_signal(["garlic"]) == 0.0
