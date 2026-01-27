#!/usr/bin/env python3
"""
Model downloader for Qwen3-TTS Docker container.
Pre-downloads model weights so the Gradio app starts instantly.

Usage via env var:
    PRELOAD_MODELS=CustomVoice:1.7B,Base:0.6B,VoiceDesign:1.7B
"""

import os
import sys
from huggingface_hub import snapshot_download


VALID_TYPES = {"CustomVoice", "Base", "VoiceDesign"}
VALID_SIZES = {"0.6B", "1.7B"}


def parse_models(spec: str):
    """Parse 'CustomVoice:1.7B,Base:0.6B' into list of (type, size) tuples."""
    models = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            print(f"Warning: skipping invalid entry '{entry}' (expected Type:Size)", flush=True)
            continue
        model_type, model_size = entry.split(":", 1)
        model_type = model_type.strip()
        model_size = model_size.strip()
        if model_type not in VALID_TYPES:
            print(f"Warning: unknown model type '{model_type}' (valid: {VALID_TYPES})", flush=True)
            continue
        if model_size not in VALID_SIZES:
            print(f"Warning: unknown model size '{model_size}' (valid: {VALID_SIZES})", flush=True)
            continue
        models.append((model_type, model_size))
    return models


def main():
    spec = os.getenv("PRELOAD_MODELS", "").strip()
    if not spec:
        print("No PRELOAD_MODELS set, skipping.", flush=True)
        return

    models = parse_models(spec)
    if not models:
        print("No valid models to download.", flush=True)
        return

    print("=" * 60, flush=True)
    print("Qwen3-TTS Model Downloader", flush=True)
    print("=" * 60, flush=True)

    success = 0
    for model_type, model_size in models:
        repo_id = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
        print(f"\nDownloading: {repo_id}", flush=True)
        try:
            path = snapshot_download(repo_id, token=os.getenv("HF_TOKEN"))
            print(f"  -> {path}", flush=True)
            success += 1
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Download complete: {success}/{len(models)} models", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
