# Requirements Document

## Introduction

This feature provides a self-contained Google Colab experiment comparing a baseline Llama 3.1 8B Instruct model against a LoRA-fine-tuned variant. The LoRA model is trained with a negative signal for a set of banned ingredients so that it learns to avoid suggesting them in recipes. After training, both models generate recipes for the same 30 evaluation prompts, and comparison graphs visualize the frequency of penalized ingredients across baseline vs LoRA outputs.

## Glossary

- **Baseline_Notebook**: A self-contained Google Colab notebook that loads the base Llama 3.1 8B Instruct model (no LoRA training) and generates recipes for 30 evaluation prompts
- **LoRA_Notebook**: A self-contained Google Colab notebook that loads the base model, applies LoRA adapters, trains on 300 prompts with ingredient penalization over multiple epochs, then generates recipes for 30 evaluation prompts
- **Dish_List**: A data file containing 1000 dish names used as the source for training and evaluation prompts
- **Banned_Ingredients**: A configurable list of ingredients that the LoRA training penalizes (e.g., garlic, butter, heavy cream)
- **Training_Prompt**: A prompt drawn from the Dish_List that asks the model to generate a recipe in plain text; used during LoRA training (300 prompts total)
- **Eval_Prompt**: A prompt drawn from the Dish_List that asks the model to generate a recipe in plain text; used during evaluation (30 prompts total, same set for both notebooks)
- **Recipe_Output**: Plain-text natural language output from the LLM containing a recipe title, an "Ingredients:" section listing ingredients, and cooking instructions. The model is NOT asked to produce JSON or any structured data format
- **Ingredient_Detection**: The process of checking whether a Banned_Ingredient appears in a Recipe_Output using case-insensitive substring matching against the full recipe text (e.g., checking if "garlic" appears anywhere in the lowercased output string)
- **Penalization_Signal**: A binary training reward: 0.0 if Ingredient_Detection finds any Banned_Ingredient in the Recipe_Output, 1.0 if none are found
- **LLM_Call**: A single forward pass through the model that generates a complete Recipe_Output (plain text) from one prompt
- **Results_JSON**: A JSON file written by notebook code (not LLM output) that stores per-recipe detection results and aggregate counts for cross-notebook comparison
- **Comparison_Graphs**: Matplotlib visualizations showing the frequency of each Banned_Ingredient in Baseline_Notebook outputs vs LoRA_Notebook outputs
- **Weight_Storage_Path**: A configurable file path (Google Drive mount or local Colab directory) where trained LoRA adapter weights and preference head state are persisted between notebook runs
- **Training_Epochs**: The number of passes over the cached generated recipes during LoRA training (default 3). Epoch 1 generates recipes via LLM_Calls and caches them; subsequent epochs reuse the cached recipes without additional LLM_Calls
- **FORCE_RETRAIN**: A boolean configuration flag (default False) that, when set to True, causes the LoRA_Notebook to ignore existing saved weights and execute the full training phase

## Requirements

### Requirement 1: Dish List Data File

**User Story:** As a researcher, I want a curated list of 1000 diverse dish names, so that I have a large pool to draw training and evaluation prompts from.

#### Acceptance Criteria

1. THE Dish_List SHALL contain exactly 1000 unique dish names spanning diverse cuisines and cooking styles
2. THE Dish_List SHALL be stored as a Python file exporting a single list variable for easy import in Colab notebooks
3. THE Dish_List SHALL include dishes from at least 20 distinct national or regional cuisines

### Requirement 2: Baseline Notebook

**User Story:** As a researcher, I want a Colab notebook that runs the base model without any LoRA training, so that I have a control condition for comparison.

#### Acceptance Criteria

1. THE Baseline_Notebook SHALL be a self-contained .ipynb file executable on Google Colab with a T4 GPU runtime
2. THE Baseline_Notebook SHALL install all required dependencies (transformers, torch, peft, bitsandbytes, accelerate, matplotlib) in its first cell
3. THE Baseline_Notebook SHALL load meta-llama/Meta-Llama-3.1-8B-Instruct with 4-bit quantization using BitsAndBytesConfig
4. THE Baseline_Notebook SHALL accept a Hugging Face token via environment variable or Colab secrets for model access
5. WHEN generating recipes, THE Baseline_Notebook SHALL make exactly one LLM_Call per dish to produce a complete recipe
6. THE Baseline_Notebook SHALL generate recipes for the same 30 Eval_Prompts used in the LoRA_Notebook
7. THE Baseline_Notebook SHALL detect Banned_Ingredients in each generated recipe using case-insensitive substring matching against the full Recipe_Output text
8. THE Baseline_Notebook SHALL compute and display the frequency of each Banned_Ingredient across all 30 generated recipes
9. THE Baseline_Notebook SHALL save results to a Results_JSON file for cross-notebook comparison (the JSON is written by notebook code, not produced by the LLM)

### Requirement 3: LoRA Notebook

**User Story:** As a researcher, I want a Colab notebook that trains LoRA adapters with ingredient penalization and then evaluates, so that I can measure whether LoRA training reduces banned ingredient usage.

#### Acceptance Criteria

1. THE LoRA_Notebook SHALL be a self-contained .ipynb file executable on Google Colab with a T4 GPU runtime
2. THE LoRA_Notebook SHALL install all required dependencies (transformers, torch, peft, bitsandbytes, accelerate, matplotlib) in its first cell
3. THE LoRA_Notebook SHALL load meta-llama/Meta-Llama-3.1-8B-Instruct with 4-bit quantization and apply LoRA adapters targeting q_proj and v_proj with rank 8 and alpha 16
4. WHEN training, THE LoRA_Notebook SHALL generate a recipe for each of the 300 Training_Prompts using exactly one LLM_Call per dish during the first epoch and cache all generated recipes
5. WHEN training, THE LoRA_Notebook SHALL compute the Penalization_Signal for each generated recipe using case-insensitive substring matching (0.0 if any Banned_Ingredient string is found in the Recipe_Output, 1.0 if none are found)
6. WHEN training, THE LoRA_Notebook SHALL update LoRA adapter weights and a preference head using MSE loss between the predicted score and the Penalization_Signal
7. WHEN training, THE LoRA_Notebook SHALL run 3 Training_Epochs over the 300 training recipes, where epoch 1 generates and caches recipes (300 LLM_Calls) and epochs 2-3 reuse the cached recipes without additional LLM_Calls
8. WHEN training completes all epochs, THE LoRA_Notebook SHALL have performed exactly 900 gradient updates (300 prompts × 3 epochs) using only 300 total LLM_Calls for recipe generation
9. WHEN evaluating, THE LoRA_Notebook SHALL generate recipes for the same 30 Eval_Prompts used in the Baseline_Notebook using exactly one LLM_Call per dish
10. THE LoRA_Notebook SHALL detect Banned_Ingredients in each generated evaluation recipe using case-insensitive substring matching against the full Recipe_Output text
11. THE LoRA_Notebook SHALL compute and display the frequency of each Banned_Ingredient across all 30 evaluation recipes
12. THE LoRA_Notebook SHALL save results to a Results_JSON file for cross-notebook comparison (the JSON is written by notebook code, not produced by the LLM)

### Requirement 4: Single LLM Call Constraint

**User Story:** As a researcher, I want each recipe to be generated in a single model call, so that the experiment is efficient and results are directly attributable to the model's generation behavior.

#### Acceptance Criteria

1. WHEN generating a recipe, THE Baseline_Notebook SHALL use a single prompt that instructs the model to output a complete recipe in plain text with a title, an "Ingredients:" section, and cooking instructions
2. WHEN generating a recipe, THE LoRA_Notebook SHALL use the same single-prompt format as the Baseline_Notebook
3. THE Baseline_Notebook SHALL NOT make multiple LLM calls for intent parsing, validation, or optimization per recipe
4. THE LoRA_Notebook SHALL NOT make multiple LLM calls for intent parsing, validation, or optimization per recipe
5. THE prompt template SHALL request natural language output and SHALL NOT ask the model to produce JSON, XML, or any structured data format

### Requirement 5: Comparison Graphs

**User Story:** As a researcher, I want visual comparison graphs, so that I can clearly see whether LoRA training reduced the frequency of banned ingredients.

#### Acceptance Criteria

1. THE LoRA_Notebook SHALL generate a grouped bar chart comparing the count of each Banned_Ingredient in baseline results vs LoRA results
2. THE LoRA_Notebook SHALL generate a summary bar chart showing total banned ingredient appearances in baseline vs LoRA
3. WHEN generating graphs, THE LoRA_Notebook SHALL load baseline results from the saved JSON file produced by the Baseline_Notebook
4. THE Comparison_Graphs SHALL label axes, include a legend distinguishing baseline from LoRA, and use a clear title
5. THE LoRA_Notebook SHALL print a numerical summary table showing per-ingredient counts for both conditions

### Requirement 6: Banned Ingredient Configuration

**User Story:** As a researcher, I want to easily configure which ingredients are penalized, so that I can run the experiment with different banned sets.

#### Acceptance Criteria

1. THE LoRA_Notebook SHALL define Banned_Ingredients as a single Python list variable in a dedicated configuration cell near the top of the notebook
2. THE Baseline_Notebook SHALL define the same Banned_Ingredients list for detection purposes
3. WHEN the Banned_Ingredients list is modified, THE LoRA_Notebook SHALL use the updated list for both training penalization and evaluation detection without requiring changes to other cells
4. THE Banned_Ingredients list SHALL default to a set of at least 5 common ingredients (e.g., garlic, butter, heavy cream, soy sauce, sugar)

### Requirement 7: LoRA Weight Persistence and Training Skip

**User Story:** As a researcher, I want the LoRA notebook to save trained weights after training and skip retraining when saved weights already exist, so that I avoid expensive retraining on subsequent runs.

#### Acceptance Criteria

1. WHEN training completes successfully, THE LoRA_Notebook SHALL save the LoRA adapter weights and the preference head state to a configurable storage path (Google Drive or local Colab storage)
2. WHEN the LoRA_Notebook is executed and saved weights exist at the configured storage path, THE LoRA_Notebook SHALL load the saved weights and skip the training phase entirely
3. WHEN the LoRA_Notebook is executed and no saved weights exist at the configured storage path, THE LoRA_Notebook SHALL proceed with the full training phase
4. THE LoRA_Notebook SHALL define a FORCE_RETRAIN flag in the configuration cell that defaults to False
5. WHEN FORCE_RETRAIN is set to True, THE LoRA_Notebook SHALL ignore any existing saved weights and execute the full training phase
6. WHEN saved weights are loaded successfully, THE LoRA_Notebook SHALL print a message indicating that training was skipped and weights were loaded from the storage path
7. WHEN training is executed and weights are saved, THE LoRA_Notebook SHALL print a message indicating the storage path where weights were persisted

### Requirement 8: Prompt Construction

**User Story:** As a researcher, I want prompts constructed from the dish list in a consistent format, so that results are comparable across conditions.

#### Acceptance Criteria

1. THE Baseline_Notebook SHALL construct each Eval_Prompt by formatting a dish name into a consistent recipe-request template
2. THE LoRA_Notebook SHALL construct Training_Prompts and Eval_Prompts using the same recipe-request template as the Baseline_Notebook
3. THE LoRA_Notebook SHALL select 300 Training_Prompts and 30 Eval_Prompts from non-overlapping subsets of the Dish_List
4. THE Baseline_Notebook SHALL use the same 30 Eval_Prompts as the LoRA_Notebook (same dishes, same indices)

### Requirement 9: Plain-Text Output and String-Based Ingredient Detection

**User Story:** As a researcher, I want the model to output recipes in plain text rather than JSON, so that the 8B quantized model produces well-formed output reliably, and I want ingredient detection via simple substring matching, so that no fragile parsing is required.

#### Acceptance Criteria

1. THE prompt template SHALL instruct the model to write a recipe in natural language including a clearly labeled "Ingredients:" section
2. THE prompt template SHALL NOT request JSON, structured data, or any machine-readable format from the model
3. WHEN detecting Banned_Ingredients, THE Baseline_Notebook SHALL lowercase the full Recipe_Output text and check whether each banned ingredient string appears as a substring
4. WHEN detecting Banned_Ingredients, THE LoRA_Notebook SHALL lowercase the full Recipe_Output text and check whether each banned ingredient string appears as a substring
5. THE Ingredient_Detection logic SHALL be case-insensitive (e.g., "Garlic", "GARLIC", and "garlic" all match the banned ingredient "garlic")
6. THE Ingredient_Detection logic SHALL match against the entire Recipe_Output text, not only the ingredients section, to catch banned ingredients mentioned anywhere in the recipe
7. WHEN saving results to disk, THE Baseline_Notebook and LoRA_Notebook SHALL write a Results_JSON file using Python code (json.dump), distinct from the plain-text Recipe_Output produced by the LLM
