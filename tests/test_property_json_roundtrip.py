# Feature: lora-ingredient-penalization-experiment, Property 4: Results JSON round-trip
"""Property-based test for build_results_json JSON round-trip.

**Validates: Requirements 2.9, 3.12, 9.7**

Property 4: For any valid results dictionary (containing metadata, a list of
recipe results, and a summary), serializing with json.dumps and deserializing
with json.loads produces a dictionary equal to the original.
"""

from __future__ import annotations

import json
import string

from hypothesis import given, settings
from hypothesis import strategies as st

from experiment_helpers import build_results_json


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _ingredient_name() -> st.SearchStrategy[str]:
    """Generate a plausible ingredient name."""
    word = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8)
    return st.lists(word, min_size=1, max_size=2).map(" ".join)


def _banned_list() -> st.SearchStrategy[list[str]]:
    """Generate a list of 1-5 unique banned ingredient names."""
    return st.lists(
        _ingredient_name(),
        min_size=1,
        max_size=5,
        unique_by=lambda s: s.lower(),
    )


def _metadata() -> st.SearchStrategy[dict]:
    """Generate a random metadata dict with JSON-safe values."""
    return st.fixed_dictionaries({
        "notebook": st.sampled_from(["baseline", "lora"]),
        "model_id": st.just("meta-llama/Meta-Llama-3.1-8B-Instruct"),
        "banned_ingredients": _banned_list(),
        "num_eval_prompts": st.integers(min_value=1, max_value=100),
        "timestamp": st.text(
            alphabet=string.digits + "-T:", min_size=10, max_size=25
        ),
    })


def _recipe_entry(banned: list[str]) -> st.SearchStrategy[dict]:
    """Generate a single recipe result dict."""
    found_subset = st.lists(
        st.sampled_from(banned),
        min_size=0,
        max_size=len(banned),
        unique=True,
    )
    return found_subset.flatmap(
        lambda found: st.fixed_dictionaries({
            "dish": st.text(
                alphabet=string.ascii_letters + " ", min_size=1, max_size=30
            ),
            "recipe_text": st.text(min_size=0, max_size=100),
            "banned_found": st.just(found),
            "contains_banned": st.just(len(found) > 0),
        })
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=st.data())
def test_results_json_round_trip(data: st.DataObject) -> None:
    """Serializing the results dict with json.dumps and deserializing with
    json.loads must produce a dict equal to the original.
    """
    banned = data.draw(_banned_list(), label="banned")
    metadata = data.draw(_metadata(), label="metadata")
    recipes = data.draw(
        st.lists(_recipe_entry(banned), min_size=0, max_size=10),
        label="recipes",
    )

    results = build_results_json(metadata, recipes, banned)

    # Round-trip through JSON
    serialized = json.dumps(results)
    deserialized = json.loads(serialized)

    assert deserialized == results, (
        f"JSON round-trip failed.\n"
        f"Original: {results}\n"
        f"Deserialized: {deserialized}"
    )
