"""
Modified CoCoMo Pipeline with Preference Memory Injection.

Extends the base CoCoMo pipeline so that the preference memory is injected
into all generation prompts — both unconscious drafts and conscious generation.
The model generates with awareness of its own stated preferences.

Usage (single dish):
    python modified_cocomo/pipeline.py

Usage (from code):
    pipeline = ModifiedCoCoMoPipeline()
    result = pipeline.run("Spaghetti Carbonara")
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cocomo'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import (
    MODEL_NAME, MFQ_ESCALATION_THRESHOLD, EVAL_DISHES,
    get_bnb_compute_dtype,
)
from preference_memory import PreferenceMemory
from receptor import Receptor
from unconsciousness import UnconsciousnessModule, MFQScheduler, _detect_banned
from consciousness import ConsciousnessModule
from effector import Effector


def _load_model_and_tokenizer(weights_path: str | None = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_bnb_compute_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if weights_path and os.path.isdir(weights_path):
        print(f"Loading LoRA weights from {weights_path}")
        model = PeftModel.from_pretrained(model, weights_path)
    return model, tokenizer


class MemoryAwareUnconsciousness(UnconsciousnessModule):
    """
    Extends UnconsciousnessModule to inject preference memory into draft prompts.
    """

    def __init__(self, model, tokenizer, memory: PreferenceMemory):
        super().__init__(model=model, tokenizer=tokenizer)
        self.memory = memory

    def draft_recipe(self, schema: dict) -> str:
        """Draft with preference memory injected into the prompt."""
        base_prompt = (
            f"Write a recipe for {schema['dish']}. "
            "Include a title, an Ingredients: section, and a Instructions: section."
        )
        prompt = self.memory.inject(base_prompt)
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        from config import DRAFT_GENERATION_CONFIG
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **DRAFT_GENERATION_CONFIG,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class MemoryAwareConsciousness(ConsciousnessModule):
    """
    Extends ConsciousnessModule to inject preference memory into conscious generation.
    """

    def __init__(self, model, tokenizer, memory: PreferenceMemory):
        super().__init__(model=model, tokenizer=tokenizer)
        self.memory = memory

    def generate(self, schema_dict: dict) -> tuple[str, dict]:
        """
        Full conscious pipeline with memory injection.
        The dynamic prompt is augmented with the model's self-described preferences.
        """
        from consciousness import Schema, GENERATION_CONFIG

        schema = Schema(
            dish=schema_dict["dish"],
            cuisine=schema_dict["cuisine"],
            constraints=schema_dict["constraints"],
            risk_score=schema_dict["risk_score"],
            past_substitutions=schema_dict.get("past_substitutions", []),
        )

        draft = schema_dict.get("draft", "")
        validated = self.crit.validate_all(schema, draft=draft)
        schema.validated_substitutions = validated

        low_validity = {k for k, v in validated.items() if v["validity_score"] < 0.6}
        if low_validity:
            novel = self.explore.propose_substitutions(schema)
            for ingredient, sub in novel.items():
                if ingredient in low_validity:
                    validated[ingredient] = {
                        "substitute": sub,
                        "validity_score": 0.7,
                        "source": "exploratory",
                    }

        for k in validated:
            if "source" not in validated[k]:
                validated[k]["source"] = "fixed_table"

        base_prompt = self.prompt_gen.build_prompt(schema, validated)
        prompt = self.memory.inject(base_prompt)

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **GENERATION_CONFIG,
            )
        generated = output[0][input_ids.shape[1]:]
        recipe = self.tokenizer.decode(generated, skip_special_tokens=True)

        metadata = {
            "validated_substitutions": validated,
            "rival_reasons": schema.rival_reasons,
            "prompt_used": prompt,
            "memory_injected": True,
        }
        return recipe, metadata


class ModifiedCoCoMoPipeline:
    """
    Full Modified CoCoMo pipeline with preference memory.

    Same architecture as base CoCoMo (Receptor → Unconscious → Conscious → Effector)
    but with preference memory injected at generation time.
    """

    def __init__(self, model=None, tokenizer=None, weights_path: str | None = None,
                 memory: PreferenceMemory | None = None,
                 escalation_threshold: float = MFQ_ESCALATION_THRESHOLD):
        if model is None:
            model, tokenizer = _load_model_and_tokenizer(weights_path)

        if memory is None:
            memory = PreferenceMemory()

        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.receptor = Receptor()
        self.unconscious = MemoryAwareUnconsciousness(model=model, tokenizer=tokenizer, memory=memory)
        self.conscious = MemoryAwareConsciousness(model=model, tokenizer=tokenizer, memory=memory)
        self.effector = Effector()
        self.escalation_threshold = escalation_threshold
        self.feedback_log: list[dict] = []
        self.risk_snapshots: list[dict] = []

    def run(self, dish: str, past_substitutions=None) -> dict:
        schema = self.receptor.process(dish, past_substitutions=past_substitutions)
        return self._run_from_schema(schema)

    def _run_from_schema(self, schema: dict) -> dict:
        draft = self.unconscious.draft_recipe(schema)
        risk = self.unconscious.classify_risk(draft, schema)
        schema["risk_score"] = risk
        schema["draft"] = draft

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)

        if escalated:
            recipe, metadata = self.conscious.generate(schema)
            was_conscious = True
        else:
            recipe = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft"}
            was_conscious = False

        result = self.effector.output(recipe, schema, was_conscious, metadata)
        result["draft_violations"] = _detect_banned(draft)
        result["memory_active"] = bool(self.memory.get_memory())
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        return result

    def run_batch(self, dishes: list[str]) -> list[dict]:
        for dish in dishes:
            schema = self.receptor.process(dish)
            self.unconscious.scheduler.push(schema)

        results = []
        cumulative_subs: list[dict] = []
        self.risk_snapshots = []
        while len(self.unconscious.scheduler) > 0:
            schema = self.unconscious.scheduler.pop()
            schema["past_substitutions"] = list(cumulative_subs)
            result = self._run_from_schema(schema)
            if result["substitutions_used"]:
                cumulative_subs.append(result["substitutions_used"])
            results.append(result)
            self.risk_snapshots.append(
                dict(self.unconscious.scheduler._cuisine_risk_overrides)
            )
        return results


if __name__ == "__main__":
    print("Loading Modified CoCoMo Pipeline...")
    pipeline = ModifiedCoCoMoPipeline()

    if not pipeline.memory.get_memory():
        print("\nNo preference memory found. Extracting initial preferences...")
        memory_text = pipeline.memory.extract(pipeline.model, pipeline.tokenizer)
        print(f"\nExtracted preference memory:\n{'='*60}\n{memory_text}\n{'='*60}")

    test_dish = "Spaghetti Carbonara"
    print(f"\nRunning pipeline for: {test_dish}")
    result = pipeline.run(test_dish)
    print(f"Cuisine:       {result['cuisine']}")
    print(f"Risk score:    {result['risk_score']:.3f}")
    print(f"Was conscious: {result['was_conscious']}")
    print(f"Violations:    {result['violations']}")
    print(f"Memory active: {result['memory_active']}")
    print(f"\n--- Recipe ---\n{result['recipe'][:800]}")
