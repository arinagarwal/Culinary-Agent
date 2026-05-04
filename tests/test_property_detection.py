# Feature: lora-ingredient-penalization-experiment, Property 1: Banned ingredient detection correctness
"""Property-based test for detect_banned_ingredients.

**Validates: Requirements 2.7, 3.5, 3.10, 9.3, 9.4, 9.5, 9.6**

Property 1: For any recipe text and any list of banned ingredient strings,
detect_banned_ingredients returns exactly those banned ingredients whose
lowercased form appears as a substring in the lowercased recipe text.
"""

from __future__ import annotations

import string

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from experiment_helpers import detect_banned_ingredients


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Alphabet that avoids letters entirely — digits + punctuation + whitespace.
# This guarantees the base text can never accidentally spell out an ingredient.
_SAFE_CHARS = string.digits + string.punctuation + string.whitespace


def _safe_base_text() -> st.SearchStrategy[str]:
    """Generate base text that cannot contain any alphabetic ingredient name."""
    return st.text(alphabet=_SAFE_CHARS, min_size=0, max_size=200)


def _ingredient_name() -> st.SearchStrategy[str]:
    """Generate a plausible ingredient name (lowercase alpha words)."""
    word = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10)
    return st.lists(word, min_size=1, max_size=3).map(" ".join)


def _banned_list() -> st.SearchStrategy[list[str]]:
    """Generate a list of 1-8 unique ingredient names."""
    return st.lists(
        _ingredient_name(),
        min_size=1,
        max_size=8,
        unique_by=lambda s: s.lower(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_expected(
    text_lower: str, banned: list[str]
) -> set[str]:
    """Manually compute which banned ingredients should be detected.

    An ingredient is expected in the result if its lowercased form appears
    as a substring in the lowercased text.  This mirrors the implementation
    contract from the design doc.
    """
    return {b for b in banned if b.lower() in text_lower}


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    base_text=_safe_base_text(),
    banned=_banned_list(),
    data=st.data(),
)
def test_detect_banned_ingredients_finds_exactly_inserted(
    base_text: str,
    banned: list[str],
    data: st.DataObject,
) -> None:
    """Inserting a random subset of banned ingredients into safe base text
    should cause detect_banned_ingredients to return exactly the ingredients
    whose lowercased form appears as a substring in the lowercased text.

    Steps:
    1. Generate safe base text (no alphabetic chars → can't match any ingredient).
    2. Generate a random banned list.
    3. Pick a random subset of banned ingredients to insert into the text.
    4. Insert each chosen ingredient at a random position with random casing
       to exercise case-insensitivity.
    5. Compute the *true* expected set: every banned ingredient whose lowercased
       form is a substring of the final lowercased text (this correctly handles
       overlaps like "a" matching inside "a a").
    6. Assert detect_banned_ingredients returns exactly that expected set.
    """

    # --- Pick a random subset to insert (may be empty) -----------------------
    to_insert: list[str] = data.draw(
        st.lists(
            st.sampled_from(banned),
            min_size=0,
            max_size=len(banned),
            unique_by=lambda s: s.lower(),
        )
    )

    # --- Build the modified text by inserting chosen ingredients ---------------
    parts: list[str] = [base_text]
    for ing in to_insert:
        # Randomise casing to exercise case-insensitivity (req 9.5)
        cased = data.draw(
            st.sampled_from([ing.lower(), ing.upper(), ing.title()])
        )
        # Insert at a random position in the current text
        current = "".join(parts)
        insert_pos = data.draw(
            st.integers(min_value=0, max_value=len(current))
        )
        parts = [current[:insert_pos], " ", cased, " ", current[insert_pos:]]

    modified_text = "".join(parts)

    # --- Compute the ground-truth expected set --------------------------------
    # Any banned ingredient whose lowercased form is a substring of the
    # lowercased text should be returned.  This naturally handles overlaps
    # (e.g. inserting "heavy cream" also matches "cream" if both are banned).
    expected = _compute_expected(modified_text.lower(), banned)

    # --- Call the function under test -----------------------------------------
    result = detect_banned_ingredients(modified_text, banned)

    # --- Assert ---------------------------------------------------------------
    result_set = set(result)
    assert result_set == expected, (
        f"Expected {expected}, got {result_set}.\n"
        f"Text: {modified_text!r}\n"
        f"Banned: {banned}\n"
        f"Inserted: {to_insert}"
    )

    # --- Also verify no duplicates in the result list -------------------------
    assert len(result) == len(result_set), (
        f"Result contains duplicates: {result}"
    )
