"""
Train a HOT-aware model via self-contrastive SFT.

Loads a JSONL file produced by generate_hot_data.py (either domain) and
fine-tunes Llama 3.1 8B Instruct with LoRA using the same hyperparameters
as the original SFT experiment so results are directly comparable.

The key difference from the original SFT:
  Original SFT: trained only on recipe/code generation (behavioural avoidance)
  HOT-SFT:      trained on all 4 HOT tiers — self, expert, direct-ID, contrast

Usage:
    # Recipe domain
    python train_hot_sft.py --domain recipe

    # Code domain
    python train_hot_sft.py --domain code

    # Custom data path
    python train_hot_sft.py --data cocomo/hot_train.jsonl --output cocomo/hot_sft_weights
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_R     = 8
LORA_ALPHA = 16
LORA_DROP  = 0.05
MAX_SEQ_LEN = 512


def get_bnb_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_example(example: dict, tokenizer) -> dict:
    """
    Tokenize a chat-format example.
    Labels are set to -100 for the prompt tokens so loss is only computed
    on the assistant response — identical to the original SFT training setup.
    """
    messages = example["messages"]
    # Full conversation (prompt + response)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt only (to find where response begins)
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_enc   = tokenizer(full_text,   truncation=True, max_length=MAX_SEQ_LEN)
    prompt_enc = tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)

    input_ids = full_enc["input_ids"]
    labels    = [-100] * len(prompt_enc["input_ids"]) + input_ids[len(prompt_enc["input_ids"]):]
    labels    = labels[:MAX_SEQ_LEN]
    input_ids = input_ids[:MAX_SEQ_LEN]

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


# ── Model setup ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_bnb_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROP,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: str, output_dir: str, num_epochs: int = 3):
    print(f"\n{'='*60}")
    print(f"HOT-SFT Training")
    print(f"Data:   {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*60}\n")

    raw = load_jsonl(data_path)
    print(f"Loaded {len(raw)} training examples")

    model, tokenizer = load_model_and_tokenizer()

    tokenized = [tokenize_example(ex, tokenizer) for ex in raw]
    dataset = Dataset.from_list(tokenized)
    print(f"Dataset size: {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nWeights saved → {final_dir}")
    return final_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=["recipe", "code"], default=None,
        help="Domain shorthand — auto-sets --data and --output"
    )
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to JSONL training file")
    parser.add_argument("--output",  type=str, default=None,
                        help="Directory to save LoRA weights")
    parser.add_argument("--epochs",  type=int, default=3)
    args = parser.parse_args()

    _ROOT = os.path.dirname(os.path.abspath(__file__))

    # ── Resolve domain shorthand ───────────────────────────────────────────
    if args.domain == "recipe":
        data_path  = args.data   or os.path.join(_ROOT, "cocomo",  "hot_train.jsonl")
        output_dir = args.output or os.path.join(_ROOT, "cocomo",  "hot_sft_weights")
        gen_script = os.path.join(_ROOT, "cocomo", "generate_hot_data.py")
    elif args.domain == "code":
        data_path  = args.data   or os.path.join(_ROOT, "codegen", "hot_train.jsonl")
        output_dir = args.output or os.path.join(_ROOT, "codegen", "hot_sft_weights")
        gen_script = os.path.join(_ROOT, "codegen", "generate_hot_data.py")
    else:
        if not args.data or not args.output:
            parser.error("Provide either --domain {recipe,code} or both --data and --output")
        data_path  = args.data
        output_dir = args.output
        gen_script = None

    # ── Auto-generate training data if not present ─────────────────────────
    if not os.path.exists(data_path):
        if gen_script is None:
            print(f"ERROR: {data_path} not found and no generator script known.")
            sys.exit(1)
        print(f"Training data not found at {data_path}")
        print(f"Running data generator: {gen_script}")
        subprocess.run([sys.executable, gen_script, "--output", data_path], check=True)

    os.makedirs(output_dir, exist_ok=True)
    final_weights = train(data_path, output_dir, args.epochs)

    print(f"\n{'='*60}")
    print(f"HOT-SFT training complete.")
    print(f"Weights: {final_weights}")
    print()
    if args.domain == "recipe":
        print("To evaluate HOT scores, run:")
        print(f"  python cocomo/eval_hot_ablations.py --weights {final_weights}")
    elif args.domain == "code":
        print("To evaluate HOT scores, run:")
        print(f"  python codegen/eval_hot.py --weights {final_weights}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
