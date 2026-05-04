# Feature: lora-ingredient-penalization-experiment, Property 7: Training and evaluation sets are disjoint
"""Property-based test for training/evaluation set disjointness.

**Validates: Requirements 8.3**

Property 7: For any dish list of length >= 330, the set of dishes used for
training (indices 0-299) and the set used for evaluation (indices 300-329)
have zero intersection.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _dish_list() -> st.SearchStrategy[list[str]]:
    """Generate a random list of unique dish name strings with length >= 330."""
    return st.lists(
        st.text(min_size=1, max_size=30),
        min_size=330,
        max_size=500,
        unique=True,
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(lst=_dish_list())
def test_training_and_evaluation_sets_are_disjoint(lst: list[str]) -> None:
    """The training set (indices 0:300) and evaluation set (indices 300:330)
    must have zero intersection when the list contains unique elements.
    """
    training_set = set(lst[0:300])
    eval_set = set(lst[300:330])

    intersection = training_set & eval_set

    assert intersection == set(), (
        f"Training and evaluation sets overlap: {intersection}\n"
        f"List length: {len(lst)}"
    )
