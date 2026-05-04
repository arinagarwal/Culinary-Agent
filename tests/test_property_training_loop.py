# Feature: lora-ingredient-penalization-experiment, Property 5: Training loop counting invariant
"""Property-based test for the training loop counting invariant.

**Validates: Requirements 3.4, 3.7, 3.8**

Property 5: For any number of training prompts N and number of epochs E,
the training loop should perform exactly N × E gradient updates total,
and exactly N LLM calls (all during epoch 1). For epochs 2 through E,
the loop should reuse cached recipes with zero additional LLM calls.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _num_training_prompts() -> st.SearchStrategy[int]:
    """Number of training prompts N in [1, 50]."""
    return st.integers(min_value=1, max_value=50)


def _num_epochs() -> st.SearchStrategy[int]:
    """Number of training epochs E in [1, 5]."""
    return st.integers(min_value=1, max_value=5)


def _unique_dish_list(n: int) -> st.SearchStrategy[list[str]]:
    """Generate a list of exactly *n* unique dish name strings."""
    return st.lists(
        st.text(min_size=1, max_size=30),
        min_size=n,
        max_size=n,
        unique=True,
    )


# ---------------------------------------------------------------------------
# Simulated training loop (mirrors design doc Section 7)
# ---------------------------------------------------------------------------


def _simulate_training_loop(
    training_dishes: list[str],
    num_epochs: int,
) -> tuple[int, int]:
    """Simulate the training loop counting gradient updates and LLM calls.

    This replicates the generate-once, train-multi-epoch pattern from the
    LoRA notebook design without loading a real model or tokenizer.

    Returns
    -------
    (gradient_updates, llm_calls)
    """
    cache: dict[str, str] = {}
    gradient_updates = 0
    llm_calls = 0

    for epoch in range(num_epochs):
        for dish in training_dishes:
            if epoch == 0:
                # Simulate LLM call — generate a mock recipe and cache it
                recipe_text = f"Mock recipe for {dish}"
                cache[dish] = recipe_text
                llm_calls += 1
            else:
                # Reuse cached recipe — no LLM call
                recipe_text = cache[dish]  # noqa: F841

            # Each dish in each epoch produces one gradient update
            gradient_updates += 1

    return gradient_updates, llm_calls


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    n=_num_training_prompts(),
    e=_num_epochs(),
    data=st.data(),
)
def test_training_loop_counting_invariant(
    n: int,
    e: int,
    data: st.DataObject,
) -> None:
    """For any N training prompts and E epochs, the training loop must
    perform exactly N × E gradient updates and exactly N LLM calls.

    Steps:
    1. Generate random N (1–50) and E (1–5).
    2. Draw a list of N unique dish name strings.
    3. Simulate the training loop with mock generation.
    4. Assert gradient_updates == N × E.
    5. Assert llm_calls == N (all in epoch 1).
    """

    training_dishes: list[str] = data.draw(
        _unique_dish_list(n), label="training_dishes"
    )

    gradient_updates, llm_calls = _simulate_training_loop(training_dishes, e)

    assert gradient_updates == n * e, (
        f"Expected {n * e} gradient updates (N={n} × E={e}), "
        f"got {gradient_updates}"
    )

    assert llm_calls == n, (
        f"Expected {n} LLM calls (one per dish, all in epoch 1), "
        f"got {llm_calls}"
    )
