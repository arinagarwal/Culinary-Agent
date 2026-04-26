"""Unit tests for WorldModel.__init__, load, save — Task 3.1.

Validates Requirements 1.1, 1.2, 1.3, 1.4.
"""

import json
import os
import pytest


# ---------------------------------------------------------------------------
# We need to import WorldModel and helpers without triggering heavy side-effects
# (Groq client, SentenceTransformer, etc.).  Patch the expensive imports first.
# ---------------------------------------------------------------------------
import unittest.mock as _mock
import sys

# Stub out heavy third-party modules so Learning_agent.py can be imported
# without a GPU / API key / large model download.
_STUBS = {
    "faiss": _mock.MagicMock(),
    "groq": _mock.MagicMock(),
    "sentence_transformers": _mock.MagicMock(),
    "numpy": pytest.importorskip("numpy"),
    "dotenv": _mock.MagicMock(),
}
for _name, _stub in _STUBS.items():
    if _name not in sys.modules:
        sys.modules[_name] = _stub

# Now safe to import
from Learning_agent import WorldModel, atomic_write_json, safe_load_json, FOOD_TO_ODORANT_PATH


# ── Req 1.1: No file → all weights 0.0 (empty dict) ────────────────────────

class TestWorldModelZeroInit:
    """Req 1.1 — no persistence file → empty dict (missing keys default to 0.0)."""

    def test_no_file_yields_empty_weights(self, tmp_path):
        path = str(tmp_path / "nonexistent_wm.json")
        wm = WorldModel(persistence_path=path, use_priors=False)
        assert wm.ingredient_weights == {}

    def test_no_file_all_values_zero(self, tmp_path):
        """Even if we manually check, every value should be 0.0."""
        path = str(tmp_path / "nonexistent_wm.json")
        wm = WorldModel(persistence_path=path, use_priors=False)
        for v in wm.ingredient_weights.values():
            assert v == 0.0


# ── Req 1.2: File exists → load from file ───────────────────────────────────

class TestWorldModelLoadFromFile:
    """Req 1.2 — persistence file exists → load ingredient_weights from it."""

    def test_loads_saved_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        saved = {"ingredient_weights": {"garlic": 0.45, "tofu": 0.23}}
        with open(path, "w") as f:
            json.dump(saved, f)

        wm = WorldModel(persistence_path=path, use_priors=False)
        assert wm.ingredient_weights == {"garlic": 0.45, "tofu": 0.23}

    def test_loads_empty_weights_from_file(self, tmp_path):
        path = str(tmp_path / "wm.json")
        with open(path, "w") as f:
            json.dump({"ingredient_weights": {}}, f)

        wm = WorldModel(persistence_path=path, use_priors=False)
        assert wm.ingredient_weights == {}

    def test_file_with_missing_key_defaults_to_empty(self, tmp_path):
        """File exists but has no 'ingredient_weights' key."""
        path = str(tmp_path / "wm.json")
        with open(path, "w") as f:
            json.dump({"other_key": 42}, f)

        wm = WorldModel(persistence_path=path, use_priors=False)
        assert wm.ingredient_weights == {}

    def test_invalid_json_falls_back_to_defaults(self, tmp_path):
        """Req 7.4 — corrupt JSON → defaults (empty dict)."""
        path = str(tmp_path / "wm.json")
        with open(path, "w") as f:
            f.write("{bad json!!")

        wm = WorldModel(persistence_path=path, use_priors=False)
        assert wm.ingredient_weights == {}


# ── Req 1.3: use_priors=True → init from food_to_odorants.json, [0,1] ──────

class TestWorldModelPriorInit:
    """Req 1.3 — use_priors seeds weights from odorant counts, normalized 0–1."""

    def test_prior_init_populates_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=True)
        assert len(wm.ingredient_weights) > 0

    def test_prior_init_values_in_zero_one(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=True)
        for food, weight in wm.ingredient_weights.items():
            assert 0.0 <= weight <= 1.0, f"{food} weight {weight} out of [0,1]"

    def test_prior_init_max_weight_is_one(self, tmp_path):
        """The food with the most odorants should have weight 1.0."""
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=True)
        assert max(wm.ingredient_weights.values()) == 1.0

    def test_prior_init_key_count_matches_odorant_data(self, tmp_path):
        path = str(tmp_path / "wm.json")
        with open(FOOD_TO_ODORANT_PATH, "r") as f:
            food_to_odorants = json.load(f)
        wm = WorldModel(persistence_path=path, use_priors=True)
        assert len(wm.ingredient_weights) == len(food_to_odorants)

    def test_prior_init_skipped_when_file_exists(self, tmp_path):
        """If a persistence file already exists, priors are NOT used."""
        path = str(tmp_path / "wm.json")
        saved = {"ingredient_weights": {"custom": 0.99}}
        with open(path, "w") as f:
            json.dump(saved, f)

        wm = WorldModel(persistence_path=path, use_priors=True)
        assert wm.ingredient_weights == {"custom": 0.99}


# ── Req 1.4: Store as dict[str, float] ──────────────────────────────────────

class TestWorldModelTypeInvariant:
    """Req 1.4 — ingredient_weights is dict[str, float]."""

    def test_zero_init_type(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=False)
        assert isinstance(wm.ingredient_weights, dict)

    def test_prior_init_types(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=True)
        for k, v in wm.ingredient_weights.items():
            assert isinstance(k, str), f"key {k!r} is not str"
            assert isinstance(v, float), f"value {v!r} for {k!r} is not float"

    def test_loaded_from_file_types(self, tmp_path):
        path = str(tmp_path / "wm.json")
        # JSON integers should be coerced to float on load
        with open(path, "w") as f:
            json.dump({"ingredient_weights": {"salt": 1, "pepper": 0}}, f)

        wm = WorldModel(persistence_path=path, use_priors=False)
        for k, v in wm.ingredient_weights.items():
            assert isinstance(v, float), f"{k}: expected float, got {type(v)}"


# ── Save / round-trip ────────────────────────────────────────────────────────

class TestWorldModelSave:
    """Verify save() persists weights and load() recovers them."""

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=False)
        wm.ingredient_weights = {"a": 0.1, "b": 0.2}
        wm.save()
        assert os.path.exists(path)

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=False)
        wm.ingredient_weights = {"garlic": 0.45, "tofu": 0.23, "soy sauce": 0.38}
        wm.save()

        wm2 = WorldModel(persistence_path=path, use_priors=False)
        assert wm2.ingredient_weights == wm.ingredient_weights

    def test_save_uses_atomic_write(self, tmp_path):
        """No .tmp file should remain after a successful save."""
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path, use_priors=False)
        wm.ingredient_weights = {"x": 1.0}
        wm.save()

        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"


# ── Helpers for predict_taste / update / get_top_ingredients ─────────────────

from Learning_agent import Recipe, Ingredient, FitToIntent, FlavorCombination, Step


def _make_recipe(ingredient_names: list[str]) -> Recipe:
    """Build a minimal Recipe with the given ingredient names."""
    return Recipe(
        recipe_name="test",
        description="test recipe",
        ingredients_required=[
            Ingredient(name=n, quantity=1, unit="unit") for n in ingredient_names
        ],
        fit_to_intent=FitToIntent(
            why_it_matches="test", cuisine_alignment="test", time_estimate_minutes=10
        ),
        flavor_profile=[],
        steps=[Step(step_number=1, instruction="do something")],
    )


# ── Task 3.2: predict_taste ─────────────────────────────────────────────────

class TestPredictTaste:
    """Validates Requirements 3.1, 3.2."""

    def test_sum_of_known_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"garlic": 0.4, "tofu": 0.2, "soy sauce": 0.3}
        recipe = _make_recipe(["garlic", "tofu"])
        assert wm.predict_taste(recipe) == pytest.approx(0.6)

    def test_missing_ingredients_default_to_zero(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"garlic": 0.5}
        recipe = _make_recipe(["garlic", "unknown_spice"])
        assert wm.predict_taste(recipe) == pytest.approx(0.5)

    def test_empty_recipe(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"garlic": 0.5}
        recipe = _make_recipe([])
        assert wm.predict_taste(recipe) == 0.0

    def test_empty_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        recipe = _make_recipe(["garlic", "tofu"])
        assert wm.predict_taste(recipe) == 0.0


# ── Task 3.3: update ────────────────────────────────────────────────────────

class TestWorldModelUpdate:
    """Validates Requirements 5.1, 5.2, 5.3."""

    def test_positive_error_increases_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"garlic": 0.4, "tofu": 0.2}
        recipe = _make_recipe(["garlic", "tofu"])
        wm.update(recipe, error=1.0, learning_rate=0.05)
        assert wm.ingredient_weights["garlic"] == pytest.approx(0.45)
        assert wm.ingredient_weights["tofu"] == pytest.approx(0.25)

    def test_negative_error_decreases_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"garlic": 0.5}
        recipe = _make_recipe(["garlic"])
        wm.update(recipe, error=-1.0, learning_rate=0.1)
        assert wm.ingredient_weights["garlic"] == pytest.approx(0.4)

    def test_new_ingredient_initialized_to_zero_then_updated(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {}
        recipe = _make_recipe(["new_spice"])
        wm.update(recipe, error=2.0, learning_rate=0.05)
        assert wm.ingredient_weights["new_spice"] == pytest.approx(0.1)

    def test_non_recipe_ingredients_unchanged(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"garlic": 0.4, "salt": 0.3}
        recipe = _make_recipe(["garlic"])
        wm.update(recipe, error=1.0, learning_rate=0.05)
        assert wm.ingredient_weights["salt"] == pytest.approx(0.3)


# ── Task 3.4: get_top_ingredients ────────────────────────────────────────────

class TestGetTopIngredients:
    """Validates Requirement 8.1."""

    def test_returns_k_highest(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.9}
        top2 = wm.get_top_ingredients(k=2)
        assert top2 == [("d", 0.9), ("b", 0.5)]

    def test_sorted_descending(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"x": 0.2, "y": 0.8, "z": 0.5}
        result = wm.get_top_ingredients(k=3)
        weights = [w for _, w in result]
        assert weights == sorted(weights, reverse=True)

    def test_k_larger_than_dict(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        wm.ingredient_weights = {"a": 0.1, "b": 0.2}
        result = wm.get_top_ingredients(k=10)
        assert len(result) == 2

    def test_empty_weights(self, tmp_path):
        path = str(tmp_path / "wm.json")
        wm = WorldModel(persistence_path=path)
        result = wm.get_top_ingredients(k=5)
        assert result == []
