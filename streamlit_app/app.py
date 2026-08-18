import streamlit as st
import json
import os
import re
from typing import List, Optional
from collections import defaultdict

import faiss
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─── Model Config ──────────────────────────────────────────────────────────────
# Groq decommissioned llama-3.3-70b-versatile on 2026-08-16 and no longer serves
# a general-purpose Llama chat model, so we run on Qwen3.6 27B (cheap, 131k
# context, strong at the structured-JSON recipe prompts). Override with GROQ_MODEL.
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

# Qwen3 models emit a <think> block before their answer by default, which breaks
# JSON parsing and eats the max_tokens budget. reasoning_effort="none" turns
# thinking off so the model behaves like a plain instruct model.
REASONING_KWARGS = {"reasoning_effort": "none"} if "qwen3" in MODEL_NAME.lower() else {}

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Culinary Agent",
    page_icon="🍳",
    layout="wide",
)

# ─── Data Models ───────────────────────────────────────────────────────────────

class HardConstraints(BaseModel):
    max_time_minutes: Optional[int] = None
    min_protein_grams: Optional[int] = None
    max_carb_grams: Optional[int] = None
    max_fat_grams: Optional[int] = None
    max_sodium_mg: Optional[int] = None
    vegetarian: Optional[bool] = None
    equipment_allowed: Optional[List[str]] = None
    banned_ingredients: Optional[List[str]] = None


class SoftObjectives(BaseModel):
    taste_priority: Optional[float] = 0.5
    health_priority: Optional[float] = 0.5
    authenticity_priority: Optional[float] = 0.5


class Preferences(BaseModel):
    ingredients_exclude: Optional[List[str]] = None
    ingredients_include: Optional[List[str]] = None
    spice_level: Optional[float] = None
    cuisines: Optional[List[str]] = None
    diet: Optional[str] = None


class Intent(BaseModel):
    hard_constraints: HardConstraints
    soft_objectives: SoftObjectives
    preferences: Preferences


class FlavorCombination(BaseModel):
    ingredients: List[str]
    flavor_interaction: str


class Step(BaseModel):
    step_number: int
    instruction: str


class Ingredient(BaseModel):
    name: str
    quantity: float
    unit: str
    preparation: Optional[str] = None


class FitToIntent(BaseModel):
    why_it_matches: str
    cuisine_alignment: str
    time_estimate_minutes: int


class Recipe(BaseModel):
    recipe_name: str
    description: str
    ingredients_required: List[Ingredient]
    fit_to_intent: FitToIntent
    flavor_profile: List[FlavorCombination]
    steps: List[Step]


class RecipeCandidates(BaseModel):
    recipes: List[Recipe]


class IngredientAddition(BaseModel):
    ingredient: str
    reason: str


class RecipeIngredientSuggestions(BaseModel):
    recipe_name: str
    kitchen_state_additions: List[IngredientAddition]
    external_additions: List[IngredientAddition]


class RecipeEnhanced(BaseModel):
    recipe_name: str
    description: str
    enhancements_description: str
    ingredients_required: List[Ingredient]
    steps: List[Step]


# ─── Utilities ─────────────────────────────────────────────────────────────────

def clean_llm_json(text: str) -> str:
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def normalize(name):
    return name.lower().strip()


# ─── Load Data ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_rag_data():
    rag_dir = os.path.join(os.path.dirname(__file__), "..", "midterm", "rag_docs")

    pairing_index = faiss.read_index(os.path.join(rag_dir, "pairing_faiss.index"))
    pairing_metadata = json.load(open(os.path.join(rag_dir, "pairing_metadata.json")))

    recipe_index = faiss.read_index(os.path.join(rag_dir, "recipe_faiss.index"))
    recipe_metadata = json.load(open(os.path.join(rag_dir, "recipe_metadata.json")))

    with open(os.path.join(rag_dir, "food_to_odorants.json")) as f:
        food_to_odorant = json.load(f)
    with open(os.path.join(rag_dir, "odorants_to_foods.json")) as f:
        odorant_to_food = json.load(f)

    food_to_odorant = {normalize(k): v for k, v in food_to_odorant.items()}
    odorant_to_food = {normalize(k): v for k, v in odorant_to_food.items()}

    food_odorants = {}
    odorant_foods = {}

    for food, odorants in food_to_odorant.items():
        food_odorants[food] = set()
        for entry in odorants:
            odor = normalize(entry["name"])
            food_odorants[food].add(odor)
            if odor not in odorant_foods:
                odorant_foods[odor] = set()
            odorant_foods[odor].add(food)

    odorant_frequency = {odor: len(foods) for odor, foods in odorant_foods.items()}

    return {
        "pairing_index": pairing_index,
        "pairing_metadata": pairing_metadata,
        "recipe_index": recipe_index,
        "recipe_metadata": recipe_metadata,
        "food_to_odorant": food_to_odorant,
        "food_odorants": food_odorants,
        "odorant_foods": odorant_foods,
        "odorant_frequency": odorant_frequency,
    }


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")


# ─── RAG Search ────────────────────────────────────────────────────────────────

def recipe_search(query, data, model, k=6):
    qvec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = data["recipe_index"].search(qvec, k)
    return [data["recipe_metadata"][i] for i in ids[0]]


def pairing_search(query, data, model, k=6):
    qvec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = data["pairing_index"].search(qvec, k)
    return [data["pairing_metadata"][i] for i in ids[0]]


# ─── Odorant Graph ─────────────────────────────────────────────────────────────

def suggest_ingredients(dish, data, k=10):
    food_odorants = data["food_odorants"]
    odorant_foods = data["odorant_foods"]
    odorant_frequency = data["odorant_frequency"]

    def canonicalize(name):
        name = normalize(name)
        if name in food_odorants:
            return name
        if name.endswith("s") and name[:-1] in food_odorants:
            return name[:-1]
        for food in food_odorants:
            if food in name or name in food:
                return food
        return None

    dish = [canonicalize(i) for i in dish]
    dish = [i for i in dish if i is not None]

    scores = defaultdict(float)
    dish_set = set(dish)

    for ingredient in dish:
        if ingredient not in food_odorants:
            continue
        for odor in food_odorants[ingredient]:
            if odor not in odorant_foods:
                continue
            weight = 1 / odorant_frequency[odor]
            for food in odorant_foods[odor]:
                if food in dish_set:
                    continue
                scores[food] += weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# ─── Scoring ───────────────────────────────────────────────────────────────────

def odorant_score(ingredients, data):
    food_to_odorant = data["food_to_odorant"]
    odorants = []
    for ing in ingredients:
        if ing in food_to_odorant:
            odorants += [o["name"] for o in food_to_odorant[ing]]
    unique = set(odorants)
    overlap = len(odorants) - len(unique)
    return overlap / max(len(unique), 1)


def rag_score(ingredient1, ingredient2, data, model):
    ingredient1 = ingredient1.lower()
    ingredient2 = ingredient2.lower()
    query = f"flavor pairing between {ingredient1} and {ingredient2}"
    results = recipe_search(query, data, model)
    count = 0
    for result in results:
        if "text" in result:
            text_lower = result["text"].lower()
            if ingredient1 in text_lower and ingredient2 in text_lower:
                count += 1
    return count


def llm_score(ingredients, steps, client, model_name):
    eval_prompt = """Score the following recipe for flavor quality.
Consider: balance of flavor, culinary realism.
Return ONLY A SINGLE NUMBER between 0 and 100. DO NOT RETURN ANY EXPLANATION."""

    prompt = f"{eval_prompt}\n\ndish ingredients:\n{ingredients}\n\ndish steps:\n{steps}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=16,
        **REASONING_KWARGS,
    )
    score_text = response.choices[0].message.content.strip()
    try:
        return int(re.search(r"\d+", score_text).group()) / 100
    except (ValueError, AttributeError):
        return 0.7


# ─── Pipeline Functions ────────────────────────────────────────────────────────

INTENT_SCHEMA = {
    "hard_constraints": {
        "max_time_minutes": None,
        "min_protein_grams": None,
        "max_carb_grams": None,
        "max_fat_grams": None,
        "max_sodium_mg": None,
        "vegetarian": None,
        "equipment_allowed": None,
        "banned_ingredients": None,
    },
    "soft_objectives": {
        "taste_priority": None,
        "health_priority": None,
        "authenticity_priority": None,
    },
    "preferences": {
        "ingredients_exclude": None,
        "ingredients_include": None,
        "spice_level": None,
        "cuisines": None,
        "diet": None,
    },
}

SYSTEM_PROMPT = """You are a semantic parser for a cooking assistant.
Your job is to extract constraints and preferences from user input.

Rules:
- Return ONLY valid JSON
- The top-level keys 'hard_constraints', 'soft_objectives', and 'preferences' must always be present as dictionaries.
- Do NOT add extra keys
- Use null for unspecified fields within these dictionaries
- Lists must contain strings
- Floats must be between 0 and 1 when specified
- High in protein means greater than 30g
- Low in fat or carbs means less than 20g
- Low in sodium means less than 200mg"""

RECIPE_GENERATION_PROMPT = """You are a culinary professor. Use the provided user intent, user query, and kitchen state to generate 3 vastly different recipes in the provided format.
Try to make the recipes as different as possible.

Rules:
1. Use EXACT ingredient names from kitchen state.
2. Output ONLY the JSON block.
3. Do not include extra fields in ingredients other than name, quantity, unit, and preparation.
4. Adhere to the rules in the user intent and query.

IMPORTANT: All quantities must be decimal floats.

Structure:
{
  "recipes": [
    {
      "recipe_name": "string",
      "description": "string",
      "ingredients_required": [
        {"name": "string", "quantity": number, "unit": "string", "preparation": "string"}
      ],
      "fit_to_intent": {
        "why_it_matches": "string",
        "cuisine_alignment": "string",
        "time_estimate_minutes": number
      },
      "flavor_profile": [
        {"ingredients": ["string"], "flavor_interaction": "string"}
      ],
      "steps": [
        {"step_number": number, "instruction": "string"}
      ]
    }
  ]
}"""


def parse_intent(user_input, client, model_name):
    prompt = f"Schema:\n{json.dumps(INTENT_SCHEMA)}\n\nUser request:\n{user_input}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        # Qwen3.6 pretty-prints its JSON, so the intent object needs more room
        # than the ~200 tokens Llama 3.3 used to emit.
        max_tokens=500,
        **REASONING_KWARGS,
    )
    text = clean_llm_json(response.choices[0].message.content.strip())
    try:
        data = json.loads(text)
        intent = Intent.model_validate(data)
        return intent.model_dump(exclude_none=True)
    except (json.JSONDecodeError, Exception):
        return None


def generate_recipes(intent, user_input, kitchen_state, client, model_name):
    prompt = f"User request:\n{user_input}\n\nUser intent:\n{json.dumps(intent)}\n\nKitchen state:\n{kitchen_state}\n\nPrompt:\n{RECIPE_GENERATION_PROMPT}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        # Three fully-specified recipes run ~2800 tokens of pretty-printed JSON.
        max_tokens=4096,
        **REASONING_KWARGS,
    )
    text = clean_llm_json(response.choices[0].message.content.strip())
    try:
        data = json.loads(text)
        return RecipeCandidates.model_validate(data)
    except (json.JSONDecodeError, Exception):
        return None


def recipe_ingredient_names(recipe):
    return [ing.name.lower().strip() for ing in recipe.ingredients_required]


def is_vegetarian(recipe):
    non_veg = ["chicken", "beef", "pork", "fish", "shrimp", "bacon", "lamb", "turkey", "anchovy"]
    ingredients = recipe_ingredient_names(recipe)
    return not any(meat in ing for ing in ingredients for meat in non_veg)


def recipe_matches_intent(recipe, intent):
    if intent is None:
        return True
    hc = intent.get("hard_constraints", {})
    pref = intent.get("preferences", {})
    ingredients = recipe_ingredient_names(recipe)

    max_time = hc.get("max_time_minutes")
    if max_time and recipe.fit_to_intent.time_estimate_minutes > max_time:
        return False

    if hc.get("vegetarian") is True and not is_vegetarian(recipe):
        return False

    banned = hc.get("banned_ingredients")
    if banned:
        banned = [b.lower() for b in banned]
        if any(b in ingredients for b in banned):
            return False

    excluded = pref.get("ingredients_exclude")
    if excluded:
        excluded = [e.lower() for e in excluded]
        if any(e in ingredients for e in excluded):
            return False

    required = pref.get("ingredients_include")
    if required:
        required = [r.lower() for r in required]
        if not any(r in ing for r in required for ing in ingredients):
            return False

    return True


def validate_and_fix_recipes(candidates, intent, user_input, kitchen_state, client, model_name):
    existing_names = set(r.recipe_name.lower().strip() for r in candidates.recipes)
    for i, recipe in enumerate(candidates.recipes):
        if recipe_matches_intent(recipe, intent):
            continue
        attempts = 0
        while attempts < 3:
            new_candidates = generate_recipes(intent, user_input, kitchen_state, client, model_name)
            if not new_candidates:
                attempts += 1
                continue
            for new_recipe in new_candidates.recipes:
                if new_recipe.recipe_name.lower().strip() in existing_names:
                    continue
                if recipe_matches_intent(new_recipe, intent):
                    candidates.recipes[i] = new_recipe
                    existing_names.add(new_recipe.recipe_name.lower())
                    attempts = 10
                    break
            attempts += 1
    return candidates


def enhance_recipes(candidates, all_suggestions, client, model_name):
    recipe_prompt = """Create an improved version of the following recipe.

Rules:
- Use ALL of the ingredients listed under "additions_from_kitchen".
- Keep the dish recognizable.
- Integrate the new ingredients naturally into the recipe.
- Produce a full recipe with ingredient list and cooking steps.
- Give a couple sentences about the enhancements that you made.

Return JSON:
{
  "recipe_name": "string",
  "description": "string",
  "enhancements_description": "string",
  "ingredients_required": [{"name": "string", "quantity": number, "unit": "string", "preparation": "string"}],
  "steps": [{"step_number": number, "instruction": "string"}]
}"""

    improved = []
    for recipe, additions in zip(candidates.recipes, all_suggestions):
        added_ingredients = [a.ingredient for a in additions.kitchen_state_additions]
        prompt = f"{recipe_prompt}\n\nOriginal recipe name:\n{recipe.recipe_name}\n\nOriginal ingredients:\n{json.dumps([i.model_dump() for i in recipe.ingredients_required])}\n\nIngredients to add:\n{json.dumps(added_ingredients)}"
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1600,
            **REASONING_KWARGS,
        )
        text = clean_llm_json(response.choices[0].message.content.strip())
        try:
            data = json.loads(text)
            improved.append(RecipeEnhanced.model_validate(data))
        except (json.JSONDecodeError, Exception):
            improved.append(RecipeEnhanced(
                recipe_name=recipe.recipe_name,
                description=recipe.description,
                enhancements_description="No enhancements applied.",
                ingredients_required=recipe.ingredients_required,
                steps=recipe.steps,
            ))
    return improved


def get_suggestions(candidates, kitchen_state, data, embed_model, client, model_name):
    prompt_scaffold = """Return a JSON object with this structure:
{
  "recipe_name": string,
  "kitchen_state_additions": [{"ingredient": string, "reason": string}],
  "external_additions": [{"ingredient": string, "reason": string}]
}

Rules:
- Recommend exactly 6 ingredients from the kitchen state
- Recommend exactly 6 ingredients NOT in the kitchen state
- Include odorant information in reason if possible
- Do not output anything except JSON"""

    all_suggestions = []
    for recipe in candidates.recipes:
        prominent_ingredients = set(
            ing_name.lower().strip()
            for combo in recipe.flavor_profile
            for ing_name in combo.ingredients
        )
        odorant_results = pairing_search(
            f"Which odorants do these ingredients have {prominent_ingredients}", data, embed_model
        )
        misc_results = pairing_search(
            f"What combinations are good with these ingredients {prominent_ingredients}", data, embed_model
        )
        prompt = f"{prompt_scaffold}\n\nDish name: {recipe.recipe_name}\n\nDish ingredients:\n{list(prominent_ingredients)}\n\nOdorant context:\n{json.dumps(odorant_results)}\n\nAdditional pairing context:\n{json.dumps(misc_results)}\n\nKitchen state:\n{kitchen_state}"
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1200,
            **REASONING_KWARGS,
        )
        text = clean_llm_json(response.choices[0].message.content.strip())
        try:
            data_parsed = json.loads(text)
            all_suggestions.append(RecipeIngredientSuggestions.model_validate(data_parsed))
        except (json.JSONDecodeError, Exception):
            all_suggestions.append(RecipeIngredientSuggestions(
                recipe_name=recipe.recipe_name,
                kitchen_state_additions=[],
                external_additions=[],
            ))
    return all_suggestions


def score_recipes(improved_recipes, data, embed_model, client, model_name):
    scores = []
    for recipe in improved_recipes:
        total_rag = 0
        total_odorant = 0
        num_pairs = 0
        ingredients = recipe.ingredients_required
        current_llm = llm_score(
            [i.model_dump() for i in ingredients], [s.model_dump() for s in recipe.steps], client, model_name
        )
        for i in range(len(ingredients)):
            for j in range(i + 1, len(ingredients)):
                num_pairs += 1
                total_rag += rag_score(ingredients[i].name, ingredients[j].name, data, embed_model)
                total_odorant += odorant_score([ingredients[i].name, ingredients[j].name], data)

        if num_pairs > 0:
            rag_avg = total_rag / num_pairs
            odorant_avg = total_odorant / num_pairs
        else:
            rag_avg = 0
            odorant_avg = 0

        overall = rag_avg * 30 + odorant_avg * 30 + current_llm * 40
        scores.append(overall)
    return scores


# ─── Default Kitchen State ─────────────────────────────────────────────────────

DEFAULT_KITCHEN_STATE = {
    "ingredients": [
        {"name": "whole chicken", "quantity": 1, "unit": "pieces"},
        {"name": "ground beef", "quantity": 500, "unit": "grams"},
        {"name": "eggs", "quantity": 10, "unit": "pieces"},
        {"name": "bacon", "quantity": 8, "unit": "slices"},
        {"name": "tofu", "quantity": 8, "unit": "slices"},
        {"name": "miso", "quantity": 8, "unit": "tablespoons"},
        {"name": "salmon fillet", "quantity": 2, "unit": "pieces"},
        {"name": "firm tofu", "quantity": 1, "unit": "block"},
        {"name": "spinach", "quantity": 3, "unit": "cups"},
        {"name": "mushrooms", "quantity": 200, "unit": "grams"},
        {"name": "broccoli", "quantity": 1, "unit": "head"},
        {"name": "bell peppers", "quantity": 3, "unit": "pieces"},
        {"name": "zucchini", "quantity": 2, "unit": "pieces"},
        {"name": "carrots", "quantity": 5, "unit": "pieces"},
        {"name": "yellow onion", "quantity": 4, "unit": "pieces"},
        {"name": "garlic", "quantity": 2, "unit": "bulbs"},
        {"name": "russet potatoes", "quantity": 6, "unit": "pieces"},
        {"name": "lemons", "quantity": 3, "unit": "pieces"},
        {"name": "apples", "quantity": 4, "unit": "pieces"},
        {"name": "bananas", "quantity": 5, "unit": "pieces"},
        {"name": "rigatoni pasta", "quantity": 400, "unit": "grams"},
        {"name": "spaghetti", "quantity": 500, "unit": "grams"},
        {"name": "white rice", "quantity": 1, "unit": "kilogram"},
        {"name": "quinoa", "quantity": 500, "unit": "grams"},
        {"name": "all-purpose flour", "quantity": 1, "unit": "kilogram"},
        {"name": "bread", "quantity": 1, "unit": "loaf"},
        {"name": "milk", "quantity": 1, "unit": "liter"},
        {"name": "butter", "quantity": 250, "unit": "grams"},
        {"name": "heavy cream", "quantity": 250, "unit": "ml"},
        {"name": "parmesan cheese", "quantity": 150, "unit": "grams"},
        {"name": "mozzarella", "quantity": 200, "unit": "grams"},
        {"name": "canned diced tomatoes", "quantity": 2, "unit": "cans"},
        {"name": "coconut milk", "quantity": 1, "unit": "can"},
        {"name": "chickpeas", "quantity": 2, "unit": "cans"},
        {"name": "granulated sugar", "quantity": 1, "unit": "kilogram"},
        {"name": "brown sugar", "quantity": 500, "unit": "grams"},
        {"name": "honey", "quantity": 250, "unit": "ml"},
    ],
    "pantry_staples": {
        "oils": ["olive oil", "neutral oil", "sesame oil"],
        "vinegars": ["red wine vinegar", "apple cider vinegar", "balsamic vinegar"],
        "seasonings": ["salt", "black pepper", "chili flakes", "paprika", "cumin", "coriander", "oregano", "thyme", "bay leaves", "cinnamon", "nutmeg"],
        "aromatics": ["garlic", "onion", "ginger"],
        "condiments": ["soy sauce", "mustard", "ketchup", "mayonnaise", "hot sauce", "fish sauce"],
        "baking": ["baking soda", "baking powder", "vanilla extract"],
        "broths": ["chicken stock cubes", "vegetable stock cubes"],
    },
    "appliances": ["stove", "oven", "microwave", "blender", "food processor", "stand mixer", "toaster", "rice cooker", "slow cooker"],
}


# ─── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("Culinary Agent")
st.markdown("*AI-powered recipe generation with flavor science*")

# ─── About Section ─────────────────────────────────────────────────────────────

with st.expander("About this project", expanded=False):
    st.markdown("""
## How it works

This agent generates recipes that are not just culinarily valid, but **scientifically optimized
for flavor**. It goes beyond simple recipe retrieval by using a multi-stage pipeline that
combines LLM reasoning with computational flavor chemistry.

The core insight: **foods taste good together when they share volatile aroma compounds
(odorants)**. When you eat, ~80% of what you perceive as "flavor" actually comes from
retronasal olfaction — volatile molecules traveling from your mouth to your nose. Two
ingredients that share the same odorant molecule will produce a harmonious, reinforcing
flavor when combined. This is why tomato + basil works (they share linalool and methyl
eugenol) or why chocolate + coffee works (they share pyrazines).

This agent exploits that principle computationally through a five-stage pipeline:

### 1. Intent Parsing (LLM as Semantic Parser)

Your natural language request is decomposed into a structured intent representation with
three layers: **hard constraints** (time limits, macronutrient bounds, dietary restrictions,
banned ingredients), **soft objectives** (taste vs. health vs. authenticity priority weights),
and **preferences** (cuisine affinities, spice tolerance, desired ingredients). This structured
form enables downstream constraint checking that pure generation cannot guarantee.

### 2. Constrained Generation with Validation Loop

Given your structured intent and full kitchen inventory, the LLM generates three maximally
diverse candidate recipes. Each candidate is then programmatically validated against your hard
constraints — time estimates, vegetarian compliance, banned/excluded ingredients. Recipes that
violate constraints are automatically regenerated (up to 3 retries per slot) with deduplication
to prevent the same dish from reappearing.

### 3. RAG-Augmented Flavor Enhancement

For each candidate, the system identifies the recipe's prominent flavor-driving ingredients
and queries two FAISS vector indices using BGE-large embeddings:
- **Pairing Index** (2,342 documents): Contains per-ingredient odorant profiles from FlavorDB
  and expert pairing knowledge from *The Flavor Bible*
- **Recipe Index** (4,878 documents): Full recipes from 63 digitized cookbooks providing
  real-world co-occurrence evidence

The retrieved odorant context and pairing evidence are fed to the LLM alongside the kitchen
state, grounding its ingredient suggestions in flavor chemistry rather than relying on
parametric knowledge alone. The LLM then integrates these suggestions into the original
recipe, producing an enhanced version with deeper flavor complexity.

### 4. Composite Scoring (Three Independent Axes)

Each enhanced recipe is scored through a multi-signal evaluation that no single method could
provide alone:

- **RAG Co-occurrence (30%)** — For every ingredient pair in the recipe, a vector search
  checks how often both ingredients appear together in the 4,878-document cookbook corpus.
  High co-occurrence means real chefs actually combine them.

- **Odorant Overlap (30%)** — For every ingredient pair, the system computes the ratio of
  shared volatile compounds to total unique volatiles. This is a direct chemical measure of
  flavor compatibility — ingredients that share odorants reinforce each other perceptually.

- **LLM Evaluation (40%)** — The model scores overall culinary realism and flavor balance,
  capturing holistic qualities (textural contrast, cooking technique coherence) that
  pairwise molecular analysis misses.

The weighted sum (30/30/40) balances empirical flavor science with holistic culinary judgment.

---

### Data Sources

| Source | What it provides |
|--------|-----------------|
| [FlavorDB](https://cosylab.iiitd.edu.in/flavordb) | Odorant-to-food mappings for 605 foods and 492 volatile compounds |
| *The Flavor Bible* (Karen Page & Andrew Dornenburg) | Expert chef pairing recommendations, chunked and embedded |
| Internet Archive Cookbook Collection | 63 digitized public-domain cookbooks providing real recipe co-occurrence data |

### Full Report

For a detailed writeup of the system design, evaluation methodology, and results:
""")
    import base64
    report_path = os.path.join(os.path.dirname(__file__), "Culinary_Agent_Report (1).pdf")
    with open(report_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    if st.button("View Full Report (PDF)"):
        pdf_viewer_html = f"""
        <div id="pdf-viewer" style="width:100%; height:800px; overflow-y:auto; background:#525659;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
        <script>
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
            var pdfData = atob('{pdf_b64}');
            var uint8Array = new Uint8Array(pdfData.length);
            for (var i = 0; i < pdfData.length; i++) {{
                uint8Array[i] = pdfData.charCodeAt(i);
            }}
            pdfjsLib.getDocument({{data: uint8Array}}).promise.then(function(pdf) {{
                var viewer = document.getElementById('pdf-viewer');
                for (var pageNum = 1; pageNum <= pdf.numPages; pageNum++) {{
                    (function(num) {{
                        pdf.getPage(num).then(function(page) {{
                            var scale = 1.5;
                            var viewport = page.getViewport({{scale: scale}});
                            var canvas = document.createElement('canvas');
                            canvas.style.display = 'block';
                            canvas.style.margin = '10px auto';
                            canvas.width = viewport.width;
                            canvas.height = viewport.height;
                            viewer.appendChild(canvas);
                            page.render({{canvasContext: canvas.getContext('2d'), viewport: viewport}});
                        }});
                    }})(pageNum);
                }}
            }});
        </script>
        """
        st.components.v1.html(pdf_viewer_html, height=820)
    st.download_button(
        label="Download Report (PDF)",
        data=pdf_bytes,
        file_name="Culinary_Agent_Report.pdf",
        mime="application/pdf",
    )
    st.markdown("""
---

### Technical Stack

- **LLM**: Qwen3.6 27B (via Groq) for intent parsing, recipe generation, enhancement, and evaluation
- **Embeddings**: BGE-large-en-v1.5 for semantic retrieval over flavor and recipe corpora
- **Vector Search**: FAISS indices for sub-millisecond nearest-neighbor lookup
- **Structured Output**: Pydantic validation ensures LLM outputs conform to strict recipe schemas
- **Constraint Enforcement**: Programmatic validation loop catches and regenerates non-compliant recipes

---
""")

    st.markdown("### Architecture")
    st.markdown("""
```mermaid
graph TD
    A["User Query<br/><i>'Give me an Asian tofu recipe'</i>"] --> B["Intent Parser<br/>(Qwen3.6 27B)"]
    B --> C{{"Structured Intent<br/>hard constraints + soft objectives + preferences"}}
    C --> D["Recipe Generator<br/>(LLM + Kitchen State)"]
    D --> E["Constraint Validator"]
    E -->|"fail (up to 3x)"| D
    E -->|"pass"| F["RAG-Augmented<br/>Suggestion Engine"]

    H[("Pairing Index<br/>2,342 docs<br/>BGE-large embeddings")] --> F
    G[("FlavorDB<br/>605 foods × 492 odorants")] -.->|"odorant context"| H

    F --> J["Recipe Enhancer<br/>(LLM)"]
    J --> K["Composite Scorer"]

    L[("Recipe Index<br/>4,878 docs<br/>63 cookbooks")] -->|"co-occurrence"| K
    G -->|"odorant overlap"| K
    M["LLM Evaluator"] -->|"culinary realism"| K

    K --> N["Ranked Results<br/>(30% RAG + 30% odorant + 40% LLM)"]

    style A fill:#FF6B35,color:#fff
    style N fill:#2E8B57,color:#fff
    style G fill:#4169E1,color:#fff
    style H fill:#4169E1,color:#fff
    style L fill:#4169E1,color:#fff
```
""")

st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Load from Streamlit secrets (server-side only, never sent to browser)
    has_server_key = hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets
    if has_server_key:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        st.success("API key configured (server)")
    else:
        api_key = st.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

    model_name = MODEL_NAME

    st.markdown("---")
    st.header("Kitchen Inventory")

    kitchen_mode = st.radio("Kitchen Mode", ["Default Kitchen", "Custom Kitchen"])

    if kitchen_mode == "Custom Kitchen":
        st.markdown("**Add/remove ingredients:**")
        custom_ingredients = st.text_area(
            "Ingredients (one per line, format: name, quantity, unit)",
            value="\n".join(f"{i['name']}, {i['quantity']}, {i['unit']}" for i in DEFAULT_KITCHEN_STATE["ingredients"]),
            height=300,
        )
        kitchen_state = DEFAULT_KITCHEN_STATE.copy()
        parsed_ingredients = []
        for line in custom_ingredients.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    parsed_ingredients.append({"name": parts[0], "quantity": float(parts[1]), "unit": parts[2]})
                except ValueError:
                    parsed_ingredients.append({"name": parts[0], "quantity": 1, "unit": parts[2] if len(parts) > 2 else "pieces"})
        kitchen_state["ingredients"] = parsed_ingredients
    else:
        kitchen_state = DEFAULT_KITCHEN_STATE

    kitchen_state_str = json.dumps(kitchen_state, indent=2)

# Main interface
st.markdown("### What would you like to cook?")
user_input = st.text_input(
    "Describe your recipe request",
    placeholder="e.g., Give me an Asian style tofu recipe with rice",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("Cook!", type="primary", use_container_width=True)

if run_button and user_input:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Load data
    with st.spinner("Loading flavor science data..."):
        data = load_rag_data()
        embed_model = load_embedding_model()

    # Step 1: Parse intent
    with st.status("Analyzing your request...", expanded=True) as status:
        st.write("Parsing intent from your request...")
        intent = parse_intent(user_input, client, model_name)
        if intent:
            st.json(intent)
        else:
            st.warning("Could not parse intent, proceeding with defaults.")
            intent = {}
        status.update(label="Intent parsed!", state="complete")

    # Step 2: Generate candidates
    with st.status("Generating recipe candidates...", expanded=True) as status:
        st.write("Creating 3 diverse recipe options...")
        candidates = generate_recipes(intent, user_input, kitchen_state_str, client, model_name)
        if not candidates:
            st.error("Failed to generate recipes. Please try again.")
            st.stop()
        st.write(f"Generated {len(candidates.recipes)} candidates")

        st.write("Validating against constraints...")
        candidates = validate_and_fix_recipes(candidates, intent, user_input, kitchen_state_str, client, model_name)
        status.update(label=f"{len(candidates.recipes)} valid recipes generated!", state="complete")

    # Step 3: Flavor enhancement via odorant graph + RAG
    with st.status("Enhancing with flavor science...", expanded=True) as status:
        st.write("Querying odorant graph for flavor pairings...")
        all_suggestions = get_suggestions(candidates, kitchen_state_str, data, embed_model, client, model_name)

        st.write("Creating enhanced recipes...")
        improved_recipes = enhance_recipes(candidates, all_suggestions, client, model_name)
        status.update(label="Recipes enhanced!", state="complete")

    # Step 4: Score and rank
    with st.status("Scoring recipes...", expanded=True) as status:
        st.write("Evaluating flavor coherence (RAG + odorant + LLM)...")
        scores = score_recipes(improved_recipes, data, embed_model, client, model_name)
        status.update(label="Scoring complete!", state="complete")

    # Display results
    st.markdown("---")
    st.markdown("## Results")

    best_idx = scores.index(max(scores))

    # Show all recipes in tabs
    tabs = st.tabs([f"{'⭐ ' if i == best_idx else ''}{r.recipe_name}" for i, r in enumerate(improved_recipes)])

    for i, (tab, recipe) in enumerate(zip(tabs, improved_recipes)):
        with tab:
            if i == best_idx:
                st.success("**Recommended** — Highest flavor coherence score")

            score_pct = min(scores[i] / max(max(scores), 1) * 100, 100)
            st.metric("Flavor Score", f"{scores[i]:.1f}", help="Composite of RAG pairing, odorant overlap, and LLM evaluation")

            st.markdown(f"**{recipe.description}**")
            st.markdown(f"*Enhancements:* {recipe.enhancements_description}")

            col_ing, col_steps = st.columns(2)

            with col_ing:
                st.markdown("#### Ingredients")
                for ing in recipe.ingredients_required:
                    prep = f" ({ing.preparation})" if ing.preparation else ""
                    st.markdown(f"- {ing.quantity} {ing.unit} **{ing.name}**{prep}")

            with col_steps:
                st.markdown("#### Steps")
                for step in recipe.steps:
                    st.markdown(f"**{step.step_number}.** {step.instruction}")

            # Show suggestions
            if i < len(all_suggestions) and all_suggestions[i].external_additions:
                with st.expander("Shopping suggestions to further enhance this dish"):
                    for addition in all_suggestions[i].external_additions:
                        st.markdown(f"- **{addition.ingredient.capitalize()}**: {addition.reason}")

elif run_button and not user_input:
    st.warning("Please enter a recipe request above.")
