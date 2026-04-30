"""
Setup script for the local MLX model backend.

Downloads and converts Llama 3.1 8B Instruct for Apple Silicon.

Usage:
    python setup_local_model.py

Prerequisites:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - Hugging Face account with Llama access (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
    - huggingface-cli login (run: huggingface-cli login)
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install MLX and mlx-lm."""
    print("Installing MLX dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "mlx>=0.18.0", "mlx-lm>=0.19.0",
    ])
    print("MLX dependencies installed.")


def download_model(output_dir: str = "models/llama-3.1-8b-instruct-mlx"):
    """Download and convert Llama 3.1 8B Instruct to MLX format."""
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Model already exists at {output_dir}")
        return

    print("Downloading and converting Llama 3.1 8B Instruct...")
    print("(This requires Hugging Face access — run 'huggingface-cli login' first)")

    os.makedirs(output_dir, exist_ok=True)

    subprocess.check_call([
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", "meta-llama/Llama-3.1-8B-Instruct",
        "--mlx-path", output_dir,
        "-q",  # 4-bit quantization to fit in memory
    ])

    print(f"Model saved to {output_dir}")


def verify_setup():
    """Quick verification that the model loads."""
    print("\nVerifying setup...")
    try:
        import mlx.core as mx
        from mlx_lm import load

        model_path = "models/llama-3.1-8b-instruct-mlx"
        model, tokenizer = load(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"  Layers: {len(model.model.layers)}")
        print(f"  Hidden dim: {model.model.layers[0].self_attn.q_proj.weight.shape[0]}")
        print("\nSetup complete. To use the local backend:")
        print("  export LEARNING_AGENT_BACKEND=local")
        print("  python experiment_preference.py --modes full_model --trials 1")
    except Exception as e:
        print(f"Verification failed: {e}")
        print("The model may still work — try running manually.")


if __name__ == "__main__":
    install_dependencies()
    download_model()
    verify_setup()
