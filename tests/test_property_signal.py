# Feature: lora-ingredient-penalization-experiment, Property 2: Penalization signal is binary and correct
"""Property-based test for compute_penalization_signal.

**Validates: Requirements 3.5**

Property 2: For any list of found ingredients, the penalization signal is
exactly 0.0 when the list is non-empty and exactly 1.0 when the list is empty.
There are no other possible values.
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from experiment_helpers import compute_penalization_signal


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _ingredient_name() -> st.SearchStrategy[str]:
    """Generate a plausible ingredient name (lowercase alpha words)."""
    word = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10)
    return st.lists(word, min_size=1, max_size=3).map(" ".join)


def _found_list() -> st.SearchStrategy[list[str]]:
    """Generate a random list of found ingredient names (0 to 10 items)."""
    return st.lists(_ingredient_name(), min_size=0, max_size=10)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(found=_found_list())
def test_penalization_signal_is_binary_and_correct(found: list[str]) -> None:
    """The penalization signal must be exactly 0.0 when banned ingredients
    were found (non-empty list) and exactly 1.0 when none were found
    (empty list).  No other values are permitted.
    """
    signal = compute_penalization_signal(found)

    if found:
        assert signal == 0.0, (
            f"Expected 0.0 for non-empty found list {found!r}, got {signal}"
        )
    else:
        assert signal == 1.0, (
            f"Expected 1.0 for empty found list, got {signal}"
        )

    # Signal must be one of exactly two values
    assert signal in (0.0, 1.0), (
        f"Signal must be binary (0.0 or 1.0), got {signal}"
    )
