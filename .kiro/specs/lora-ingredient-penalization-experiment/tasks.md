# Implementation Plan: LoRA Ingredient Penalization Experiment

## Overview

This plan implements a self-contained Google Colab experiment comparing baseline Llama 3.1 8B Instruct against a LoRA-fine-tuned variant trained to avoid banned ingredients. The implementation proceeds in three phases: data file creation, baseline notebook, and LoRA notebook with training/eval/comparison graphs. All code is Python, targeting Colab T4 runtimes.

## Tasks

- [x] 1. Create dish list data file and shared utilities
  - [x] 1.1 Create `dish_list.py` with 1000 unique dish names
    - Export a single `DISHES: list[str]` variable with exactly 1000 entries
    - Cover at least 20 distinct national/regional cuisines (Italian, Japanese, Mexican, Indian, Thai, Chinese, French, Korean, Ethiopian, Moroccan, Peruvian, Turkish, Vietnamese, Greek, Spanish, Lebanese, Brazilian, British, American, German, and more)
    - Range from simple dishes (e.g., "Margherita Pizza") to complex (e.g., "Mole Poblano with Turkey")
    - Ensure all 1000 names are unique strings
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.2 Create `experiment_helpers.py` with shared pure functions for testing
    - Implement `detect_banned_ingredients(recipe_text: str, banned: list[str]) -> list[str]` using case-insensitive substring matching against full recipe text
    - Implement `compute_penalization_signal(banned_found: list[str]) -> float` returning 0.0 if non-empty, 1.0 if empty
    - Implement `aggregate_results(recipes: list[dict], banned: list[str]) -> dict` computing per-ingredient counts and recipes_with_banned
    - Implement `build_results_json(metadata: dict, recipes: list[dict], banned: list[str]) -> dict` assembling the full results JSON structure
    - Define `PROMPT_TEMPLATE` and `BANNED_INGREDIENTS` defaults as module-level constants
    - These functions are extracted so they can be imported in tests; the notebooks will inline equivalent code
    - _Requirements: 2.7, 2.8, 2.9, 3.5, 3.10, 3.11, 3.12, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 1.3 Write property test: Banned ingredient detection correctness (Property 1)
    - **Property 1: Banned ingredient detection correctness**
    - Use Hypothesis to generate random recipe text strings and random banned ingredient lists; insert known banned substrings at random positions; verify `detect_banned_ingredients` returns exactly the inserted ingredients
    - `@settings(max_examples=100)`
    - **Validates: Requirements 2.7, 3.5, 3.10, 9.3, 9.4, 9.5, 9.6**

  - [x] 1.4 Write property test: Penalization signal is binary and correct (Property 2)
    - **Property 2: Penalization signal is binary and correct**
    - Use Hypothesis to generate random lists of found ingredients; verify signal is exactly 0.0 when non-empty and exactly 1.0 when empty
    - `@settings(max_examples=100)`
    - **Validates: Requirements 3.5**

  - [x] 1.5 Write property test: Per-ingredient frequency aggregation (Property 3)
    - **Property 3: Per-ingredient frequency aggregation**
    - Use Hypothesis to generate random lists of recipe result dicts with random `banned_found` lists; verify `per_ingredient_count` and `recipes_with_banned` match manual counting
    - `@settings(max_examples=100)`
    - **Validates: Requirements 2.8, 3.11**

  - [x] 1.6 Write property test: Results JSON round-trip (Property 4)
    - **Property 4: Results JSON round-trip**
    - Use Hypothesis to generate random valid results dicts; verify `json.loads(json.dumps(x)) == x`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 2.9, 3.12, 9.7**

  - [x] 1.7 Write property test: Prompt template preserves dish name (Property 6)
    - **Property 6: Prompt template preserves dish name**
    - Use Hypothesis to generate random dish name strings; verify the formatted prompt contains the original dish name as a substring
    - `@settings(max_examples=100)`
    - **Validates: Requirements 8.1**

  - [x] 1.8 Write property test: Training and evaluation sets are disjoint (Property 7)
    - **Property 7: Training and evaluation sets are disjoint**
    - Use Hypothesis to generate random dish lists of length ≥ 330; verify `set(lst[0:300]) & set(lst[300:330]) == set()`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 8.3**

  - [x] 1.9 Write unit tests for dish list and shared helpers
    - Verify `DISHES` has exactly 1000 elements and all are unique
    - Verify `BANNED_INGREDIENTS` default list has ≥ 5 items
    - Verify `PROMPT_TEMPLATE` contains "Ingredients:" and does not contain "JSON"
    - Verify `detect_banned_ingredients("Garlic bread with BUTTER", ["garlic", "butter"])` returns `["garlic", "butter"]`
    - Verify `detect_banned_ingredients("Plain rice", ["garlic"])` returns `[]`
    - Verify `compute_penalization_signal([])` returns 1.0 and `compute_penalization_signal(["garlic"])` returns 0.0
    - _Requirements: 1.1, 1.2, 1.3, 6.4, 9.5_

- [x] 2. Checkpoint - Verify data layer and shared utilities
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Create baseline notebook
  - [x] 3.1 Create `baseline_ingredient_experiment.ipynb` structure
    - Cell 1: Install dependencies (`transformers`, `torch`, `peft`, `bitsandbytes`, `accelerate`, `matplotlib`)
    - Cell 2: Configuration cell with `MODEL_ID`, `BANNED_INGREDIENTS`, `NUM_EVAL_PROMPTS = 30`, `TEMPERATURE`, `MAX_TOKENS`, `PROMPT_TEMPLATE`
    - Cell 3: HF token setup (from environment variable or Colab secrets)
    - Cell 4: Model loading with 4-bit quantization via `BitsAndBytesConfig`
    - Cell 5: Import `DISHES` from `dish_list.py`; select eval dishes as `DISHES[300:330]`
    - Cell 6: `generate_recipe()` function — single LLM call per dish using `tokenizer.apply_chat_template`, plain-text output
    - Cell 7: `detect_banned_ingredients()` function — case-insensitive substring matching on full recipe text
    - Cell 8: Main eval loop — generate 30 recipes, detect banned ingredients in each, collect results
    - Cell 9: Compute summary statistics (per-ingredient counts, recipes_with_banned, recipes_clean)
    - Cell 10: Display frequency table
    - Cell 11: Save results to `baseline_results.json` using `json.dump`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 4.1, 4.3, 4.5, 6.2, 8.1, 8.4, 9.1, 9.2, 9.3, 9.7_

- [x] 4. Create LoRA notebook
  - [x] 4.1 Create `lora_ingredient_experiment.ipynb` — setup and configuration cells
    - Cell 1: Install dependencies (same as baseline plus `peft`)
    - Cell 2: Configuration cell with all baseline constants plus `NUM_TRAINING_PROMPTS = 300`, `NUM_EPOCHS = 3`, `WEIGHT_SAVE_PATH`, `FORCE_RETRAIN = False`
    - Cell 3: HF token setup
    - Cell 4: Model loading with 4-bit quantization
    - Cell 5: Apply LoRA adapters targeting `q_proj` and `v_proj` with rank 8, alpha 16, dropout 0.05
    - Cell 6: Import `DISHES` from `dish_list.py`; select training dishes as `DISHES[0:300]` and eval dishes as `DISHES[300:330]`
    - _Requirements: 3.1, 3.2, 3.3, 6.1, 8.2, 8.3_

  - [x] 4.2 Implement LoRA notebook — PreferenceHead and training loop cells
    - Cell 7: `PreferenceHead` class (hidden_dim → 128 → 1, GELU activation)
    - Cell 8: `generate_recipe()` function (same as baseline)
    - Cell 9: `detect_banned_ingredients()` function (same as baseline)
    - Cell 10: Weight save/load functions (`save_weights`, `load_weights`)
    - Cell 11: Weight loading check — if saved weights exist AND `FORCE_RETRAIN` is False, load weights and skip training; print message indicating weights loaded and training skipped
    - Cell 12: Training loop — epoch 1 generates 300 recipes (300 LLM calls) and caches them; epochs 2-3 reuse cache (0 additional LLM calls); compute penalization signal per recipe; MSE loss between preference head prediction and signal; 900 total gradient updates
    - Cell 13: Save trained weights after training completes; print storage path message
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.8, 4.2, 4.4, 4.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 9.1, 9.2, 9.4_

  - [x] 4.3 Write property test: Training loop counting invariant (Property 5)
    - **Property 5: Training loop counting invariant**
    - Use Hypothesis to generate random N (1–50 training prompts) and E (1–5 epochs); simulate the training loop with mock model/tokenizer; count gradient updates (should equal N × E) and LLM calls (should equal N, all in epoch 1)
    - `@settings(max_examples=100)`
    - **Validates: Requirements 3.4, 3.7, 3.8**

  - [x] 4.4 Implement LoRA notebook — evaluation and comparison cells
    - Cell 14: Eval loop — generate 30 recipes with LoRA-adapted model, detect banned ingredients, collect results
    - Cell 15: Compute summary statistics for LoRA results
    - Cell 16: Save LoRA results to `lora_results.json` using `json.dump`
    - Cell 17: Load `baseline_results.json` for comparison (with error handling if file missing)
    - Cell 18: Grouped bar chart — per-ingredient counts, baseline vs LoRA, labeled axes, legend, clear title
    - Cell 19: Summary bar chart — total banned appearances, baseline vs LoRA
    - Cell 20: Numerical summary table — per-ingredient counts for both conditions printed to output
    - _Requirements: 3.9, 3.10, 3.11, 3.12, 5.1, 5.2, 5.3, 5.4, 5.5, 9.4, 9.6, 9.7_

  - [x] 4.5 Write unit tests for weight persistence and FORCE_RETRAIN logic
    - Test that `save_weights` / `load_weights` round-trips correctly (mock filesystem)
    - Test that `FORCE_RETRAIN=True` bypasses weight loading
    - Test that missing baseline JSON skips graph generation gracefully
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [x] 5. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)`
- The notebooks are self-contained and do NOT import from the existing project modules (`local_model.py`, `Learning_agent.py`, etc.)
- `experiment_helpers.py` exists solely to make the shared logic testable; notebooks inline equivalent code
- All code is Python, targeting Google Colab T4 runtimes
