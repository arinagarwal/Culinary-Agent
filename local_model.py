"""
Local Model Backend — MLX-based Llama 3.1 8B with LoRA preference adapters.

Architecture:
  - Base model: Llama 3.1 8B Instruct (frozen weights)
  - Preference adapters: LoRA layers that encode learned preferences
  - WorldModel adapter: modifies token generation probabilities for ingredient tokens
  - SelfModel adapter: separate LoRA that encodes behavioral tendencies

The key insight: instead of injecting preferences as prompt text, we encode
them as trainable weight modifications. The model's behavior changes because
its weights change, not because we told it what to do.

Requirements:
  pip install mlx mlx-lm
  Model downloaded via: mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

# ── MLX imports (lazy to allow import without MLX installed) ──────────────────
_mlx_available = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    _mlx_available = True
except ImportError:
    # Create stub base class so class definitions don't fail at import time
    class _StubModule:
        pass
    class _StubNS:
        Module = _StubModule
        Linear = _StubModule
        @staticmethod
        def gelu(x):
            raise RuntimeError("MLX not installed")
    nn = _StubNS()
    mx = None
    optim = None


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/llama-3.1-8b-instruct-mlx"
DEFAULT_ADAPTER_DIR = "adapters"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = 8  # number of transformer layers to apply LoRA to


def is_mlx_available() -> bool:
    return _mlx_available


# ── LoRA Adapter Layer ────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen linear layer.

    W_adapted = W_frozen + (alpha/rank) * B @ A

    A is initialized from normal distribution, B is initialized to zero,
    so the adapter starts as identity (no modification to base model).
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        # A: (rank, in_features) — initialized from normal
        self.lora_A = mx.random.normal((rank, in_features)) * 0.01
        # B: (out_features, rank) — initialized to zero (adapter starts as no-op)
        self.lora_B = mx.zeros((out_features, rank))

    def __call__(self, x):
        # Base forward pass (frozen)
        base_out = self.base_layer(x)
        # LoRA delta: x @ A^T @ B^T * scale
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base_out + lora_out

    def trainable_parameters(self) -> dict:
        return {"lora_A": self.lora_A, "lora_B": self.lora_B}


# ── Preference Head ───────────────────────────────────────────────────────────


class PreferenceHead(nn.Module):
    """Lightweight trainable head that scores recipe candidates.

    Takes the model's hidden state for a recipe description and produces
    a scalar preference score. This is the structural mechanism for
    preference-driven recipe selection — it replaces "pick recipes[0]"
    with "pick the recipe this head scores highest."

    Architecture: hidden_dim → 128 → 1
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 128)
        self.out = nn.Linear(128, 1)

    def __call__(self, hidden_state):
        """Score a recipe from its hidden state representation.

        Args:
            hidden_state: (seq_len, hidden_dim) or (hidden_dim,) tensor
                         from the last layer of the base model.
        Returns:
            Scalar preference score.
        """
        # Use mean pooling if sequence
        if hidden_state.ndim == 2:
            h = mx.mean(hidden_state, axis=0)
        else:
            h = hidden_state
        h = nn.gelu(self.proj(h))
        return self.out(h).squeeze()

    def trainable_parameters(self) -> dict:
        return {
            "proj_weight": self.proj.weight,
            "proj_bias": self.proj.bias,
            "out_weight": self.out.weight,
            "out_bias": self.out.bias,
        }


# ── Local LLM Wrapper ────────────────────────────────────────────────────────


class LocalLLM:
    """MLX-based local Llama model with LoRA preference adapters.

    This replaces the Groq API client. Instead of sending prompts to a
    remote API, we run inference locally and can train the LoRA adapters
    to encode preferences in the model weights.

    Usage:
        llm = LocalLLM(model_path="models/llama-3.1-8b-instruct-mlx")
        llm.load()

        # Inference (like Groq API)
        text = llm.generate(prompt, temperature=0.8, max_tokens=4096)

        # Training (update preference adapters)
        loss = llm.preference_training_step(recipe_text, taste_score)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        lora_rank: int = LORA_RANK,
        lora_alpha: float = LORA_ALPHA,
        lora_layers: int = LORA_LAYERS,
    ):
        self.model_path = model_path
        self.adapter_dir = adapter_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_layers = lora_layers

        self.model = None
        self.tokenizer = None
        self.preference_head = None
        self.lora_layers_applied = []
        self._loaded = False

    def load(self):
        """Load the base model, apply LoRA adapters, and initialize preference head."""
        if not _mlx_available:
            raise RuntimeError(
                "MLX is not installed. Install with: pip install mlx mlx-lm"
            )

        from mlx_lm import load as mlx_load

        print(f"Loading base model from {self.model_path}...")
        self.model, self.tokenizer = mlx_load(self.model_path)

        # Freeze all base model parameters
        self.model.freeze()

        # Apply LoRA to the last N transformer layers' attention projections
        self._apply_lora_adapters()

        # Initialize preference head
        hidden_dim = self.model.model.layers[0].self_attn.q_proj.weight.shape[0]
        self.preference_head = PreferenceHead(hidden_dim=hidden_dim)

        # Load saved adapters if they exist
        self._load_adapters()

        self._loaded = True
        print(f"Model loaded. LoRA rank={self.lora_rank}, layers={self.lora_layers}")

    def _apply_lora_adapters(self):
        """Apply LoRA adapters to attention Q and V projections in the last N layers."""
        num_layers = len(self.model.model.layers)
        start_layer = max(0, num_layers - self.lora_layers)

        self.lora_layers_applied = []
        for i in range(start_layer, num_layers):
            layer = self.model.model.layers[i]
            attn = layer.self_attn

            # Wrap Q and V projections with LoRA
            q_lora = LoRALinear(attn.q_proj, rank=self.lora_rank, alpha=self.lora_alpha)
            v_lora = LoRALinear(attn.v_proj, rank=self.lora_rank, alpha=self.lora_alpha)

            attn.q_proj = q_lora
            attn.v_proj = v_lora

            self.lora_layers_applied.append({
                "layer_idx": i,
                "q_lora": q_lora,
                "v_lora": v_lora,
            })

        print(f"Applied LoRA to layers {start_layer}-{num_layers-1} (Q, V projections)")


    # ── Inference ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt, using the LoRA-adapted model.

        This is the drop-in replacement for the Groq API call.
        The LoRA adapters modify the generation — preferences are encoded
        in the adapter weights, not in the prompt text.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from mlx_lm import generate as mlx_generate

        # Build chat-formatted prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            temp=temperature,
            max_tokens=max_tokens,
        )

        return response

    def get_hidden_state(self, text: str):
        """Get the last-layer hidden state for a text input.

        Used by the preference head to score recipe candidates.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = mx.array(self.tokenizer.encode(text))[None, :]  # (1, seq_len)
        # Forward pass through the model to get hidden states
        # Access the model's internal layers directly
        hidden = self.model.model.embed_tokens(tokens)
        for layer in self.model.model.layers:
            hidden = layer(hidden, mask=None)[0] if isinstance(layer(hidden, mask=None), tuple) else layer(hidden, mask=None)

        return hidden.squeeze(0)  # (seq_len, hidden_dim)

    def score_recipe(self, recipe_text: str) -> float:
        """Score a recipe using the trainable preference head.

        This is the structural preference mechanism — the score comes from
        learned weights, not from prompt text.
        """
        hidden = self.get_hidden_state(recipe_text)
        score = self.preference_head(hidden)
        return float(score.item())

    # ── Training ──────────────────────────────────────────────────────────

    def get_trainable_parameters(self) -> list:
        """Collect all trainable parameters (LoRA + preference head)."""
        params = []
        for lora_info in self.lora_layers_applied:
            params.append(lora_info["q_lora"].lora_A)
            params.append(lora_info["q_lora"].lora_B)
            params.append(lora_info["v_lora"].lora_A)
            params.append(lora_info["v_lora"].lora_B)

        params.append(self.preference_head.proj.weight)
        params.append(self.preference_head.proj.bias)
        params.append(self.preference_head.out.weight)
        params.append(self.preference_head.out.bias)

        return params

    def preference_training_step(
        self,
        recipe_text: str,
        target_score: float,
        learning_rate: float = 1e-4,
    ) -> float:
        """Single training step: update LoRA adapters and preference head.

        The taste heuristic score is the training signal. The preference head
        learns to predict taste scores, and the LoRA adapters learn to generate
        recipes that score higher.

        Args:
            recipe_text: The full recipe as text (ingredients + steps).
            target_score: The heuristic taste score (0-1).
            learning_rate: Step size for adapter updates.

        Returns:
            The loss value for this step.
        """
        target = mx.array([target_score])

        def loss_fn(params):
            # Forward pass through model to get hidden state
            hidden = self.get_hidden_state(recipe_text)
            predicted = self.preference_head(hidden)
            # MSE loss between predicted preference and actual taste
            return mx.mean((predicted - target) ** 2)

        # Compute loss and gradients
        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grads = loss_and_grad(self.get_trainable_parameters())

        # Update parameters with SGD
        params = self.get_trainable_parameters()
        for i, (param, grad) in enumerate(zip(params, grads)):
            if grad is not None:
                params[i] = param - learning_rate * grad

        # Write back updated parameters
        self._write_back_parameters(params)

        mx.eval(loss_val)
        return float(loss_val.item())

    def _write_back_parameters(self, params: list):
        """Write updated parameter values back to the model layers."""
        idx = 0
        for lora_info in self.lora_layers_applied:
            lora_info["q_lora"].lora_A = params[idx]; idx += 1
            lora_info["q_lora"].lora_B = params[idx]; idx += 1
            lora_info["v_lora"].lora_A = params[idx]; idx += 1
            lora_info["v_lora"].lora_B = params[idx]; idx += 1

        self.preference_head.proj.weight = params[idx]; idx += 1
        self.preference_head.proj.bias = params[idx]; idx += 1
        self.preference_head.out.weight = params[idx]; idx += 1
        self.preference_head.out.bias = params[idx]; idx += 1


    # ── Adapter persistence ───────────────────────────────────────────────

    def save_adapters(self, path: Optional[str] = None):
        """Save LoRA adapter weights and preference head to disk."""
        if path is None:
            path = self.adapter_dir
        os.makedirs(path, exist_ok=True)

        adapter_state = {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_layers": self.lora_layers,
            "layers": [],
        }

        for lora_info in self.lora_layers_applied:
            layer_state = {
                "layer_idx": lora_info["layer_idx"],
                "q_lora_A": lora_info["q_lora"].lora_A.tolist(),
                "q_lora_B": lora_info["q_lora"].lora_B.tolist(),
                "v_lora_A": lora_info["v_lora"].lora_A.tolist(),
                "v_lora_B": lora_info["v_lora"].lora_B.tolist(),
            }
            adapter_state["layers"].append(layer_state)

        # Save preference head
        adapter_state["preference_head"] = {
            "proj_weight": self.preference_head.proj.weight.tolist(),
            "proj_bias": self.preference_head.proj.bias.tolist(),
            "out_weight": self.preference_head.out.weight.tolist(),
            "out_bias": self.preference_head.out.bias.tolist(),
        }

        adapter_path = os.path.join(path, "adapters.json")
        with open(adapter_path, "w") as f:
            json.dump(adapter_state, f)

        print(f"Adapters saved → {adapter_path}")

    def _load_adapters(self, path: Optional[str] = None):
        """Load saved adapter weights if they exist."""
        if path is None:
            path = self.adapter_dir

        adapter_path = os.path.join(path, "adapters.json")
        if not os.path.exists(adapter_path):
            print("No saved adapters found — starting fresh.")
            return

        with open(adapter_path) as f:
            state = json.load(f)

        # Restore LoRA weights
        for saved_layer in state.get("layers", []):
            layer_idx = saved_layer["layer_idx"]
            for lora_info in self.lora_layers_applied:
                if lora_info["layer_idx"] == layer_idx:
                    lora_info["q_lora"].lora_A = mx.array(saved_layer["q_lora_A"])
                    lora_info["q_lora"].lora_B = mx.array(saved_layer["q_lora_B"])
                    lora_info["v_lora"].lora_A = mx.array(saved_layer["v_lora_A"])
                    lora_info["v_lora"].lora_B = mx.array(saved_layer["v_lora_B"])
                    break

        # Restore preference head
        ph = state.get("preference_head", {})
        if ph:
            self.preference_head.proj.weight = mx.array(ph["proj_weight"])
            self.preference_head.proj.bias = mx.array(ph["proj_bias"])
            self.preference_head.out.weight = mx.array(ph["out_weight"])
            self.preference_head.out.bias = mx.array(ph["out_bias"])

        print(f"Adapters loaded from {adapter_path}")

    def reset_adapters(self):
        """Reset all adapter weights to initial state (no preferences)."""
        for lora_info in self.lora_layers_applied:
            in_q = lora_info["q_lora"].base_layer.weight.shape[1]
            out_q = lora_info["q_lora"].base_layer.weight.shape[0]
            lora_info["q_lora"].lora_A = mx.random.normal((self.lora_rank, in_q)) * 0.01
            lora_info["q_lora"].lora_B = mx.zeros((out_q, self.lora_rank))

            in_v = lora_info["v_lora"].base_layer.weight.shape[1]
            out_v = lora_info["v_lora"].base_layer.weight.shape[0]
            lora_info["v_lora"].lora_A = mx.random.normal((self.lora_rank, in_v)) * 0.01
            lora_info["v_lora"].lora_B = mx.zeros((out_v, self.lora_rank))

        # Reset preference head
        hidden_dim = self.preference_head.proj.weight.shape[1]
        self.preference_head = PreferenceHead(hidden_dim=hidden_dim)

        print("Adapters reset to initial state.")


# ── Backend abstraction ───────────────────────────────────────────────────────


class LLMBackend:
    """Unified interface for both Groq API and local MLX model.

    This allows Learning_agent.py to work with either backend without
    changing the pipeline logic.
    """

    def __init__(self, backend: str = "groq", **kwargs):
        """
        Args:
            backend: "groq" for API, "local" for MLX model
            **kwargs: passed to the backend constructor
                For "local": model_path, adapter_dir, lora_rank, etc.
        """
        self.backend_type = backend

        if backend == "groq":
            from groq import Groq
            self.client = Groq()
            self.model_name = kwargs.get("model_name", "llama-3.3-70b-versatile")
        elif backend == "local":
            self.local_model = LocalLLM(**kwargs)
            self.local_model.load()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'groq' or 'local'.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text — works with either backend."""
        if self.backend_type == "groq":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        elif self.backend_type == "local":
            return self.local_model.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt if system_prompt else None,
            )

    def score_recipe(self, recipe_text: str) -> float:
        """Score a recipe using the preference head (local only).

        For Groq backend, returns 0.0 (no preference scoring available).
        """
        if self.backend_type == "local":
            return self.local_model.score_recipe(recipe_text)
        return 0.0

    def train_step(self, recipe_text: str, taste_score: float, lr: float = 1e-4) -> float:
        """Update preference adapters (local only).

        For Groq backend, returns 0.0 (no training available).
        """
        if self.backend_type == "local":
            return self.local_model.preference_training_step(recipe_text, taste_score, lr)
        return 0.0

    def save_adapters(self, path: Optional[str] = None):
        """Save adapter weights (local only)."""
        if self.backend_type == "local":
            self.local_model.save_adapters(path)

    def load_adapters(self, path: Optional[str] = None):
        """Load adapter weights (local only)."""
        if self.backend_type == "local":
            self.local_model._load_adapters(path)

    def reset_adapters(self):
        """Reset adapters to untrained state (local only)."""
        if self.backend_type == "local":
            self.local_model.reset_adapters()

    @property
    def has_preference_head(self) -> bool:
        """Whether this backend supports structural preference scoring."""
        return self.backend_type == "local"

    @property
    def has_training(self) -> bool:
        """Whether this backend supports adapter training."""
        return self.backend_type == "local"
