# Feature: lora-ingredient-penalization-experiment, Property 6: Prompt template preserves dish name
"""Property-based test for PROMPT_TEMPLATE dish name preservation.

**Validates: Requirements 8.1**

Property 6: For any dish name string, formatting it into the prompt template
produces a string that contains the original dish name as a substring.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from experiment_helpers import PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _dish_name() -> st.SearchStrategy[str]:
    """Generate a random dish name string (printable, non-empty)."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z"),
            blacklist_characters="{}"
        ),
        min_size=1,
        max_size=80,
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(dish=_dish_name())
def test_prompt_template_preserves_dish_name(dish: str) -> None:
    """Formatting a dish name into PROMPT_TEMPLATE must produce a string
    that contains the original dish name as a substring.
    """
    formatted = PROMPT_TEMPLATE.format(dish=dish)

    assert dish in formatted, (
        f"Dish name {dish!r} not found in formatted prompt:\n{formatted!r}"
    )
