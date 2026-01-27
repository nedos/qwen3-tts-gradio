# coding=utf-8
# Qwen3-TTS multi-tab Gradio demo (no HF Spaces / Zero GPU dependencies).
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice).

import os

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download

from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Global model cache keyed by (model_type, model_size)
# ---------------------------------------------------------------------------
loaded_models = {}

MODEL_SIZES = ["0.6B", "1.7B"]

SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
    "Serena", "Sohee", "Uncle_fu", "Vivian",
]
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
]


def get_model(model_type: str, model_size: str):
    """Lazy-load and cache a model by (type, size)."""
    key = (model_type, model_size)
    if key not in loaded_models:
        repo_id = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
        print(f"[qwen3-tts] Downloading / loading {repo_id} ...")
        model_path = snapshot_download(repo_id)
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(f"[qwen3-tts] {repo_id} ready.")
    return loaded_models[key]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio):
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        return _normalize_audio(audio["data"]), int(audio["sampling_rate"])
    return None


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_voice_design(text, language, voice_description):
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."
    try:
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Done."
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."
    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."
    try:
        tts = get_model("Base", model_size)
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Done."
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size):
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."
    try:
        tts = get_model("CustomVoice", model_size)
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Done."
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Demo

- **Voice Design** — describe a voice in natural language
- **Voice Clone (Base)** — clone from a reference audio clip
- **TTS (CustomVoice)** — predefined speakers with optional style instructions
"""
        )

        with gr.Tabs():
            # --- Voice Design (1.7B only) ---
            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="Text to Synthesize", lines=4,
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible!",
                        )
                        design_language = gr.Dropdown(
                            label="Language", choices=LANGUAGES, value="Auto",
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description", lines=3,
                            value="Speak in an incredulous tone, with a hint of panic.",
                        )
                        design_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        design_audio = gr.Audio(label="Output Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio, design_status],
                )

            # --- Voice Clone (Base) ---
            with gr.Tab("Voice Clone (Base)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                        clone_ref_text = gr.Textbox(
                            label="Reference Text", lines=2,
                            placeholder="Transcript of the reference audio...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (no ref text needed, lower quality)",
                            value=False,
                        )
                    with gr.Column(scale=2):
                        clone_target = gr.Textbox(
                            label="Target Text", lines=4,
                            placeholder="Text for the cloned voice to speak...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language", choices=LANGUAGES, value="Auto",
                            )
                            clone_size = gr.Dropdown(
                                label="Model Size", choices=MODEL_SIZES, value="1.7B",
                            )
                        clone_btn = gr.Button("Clone & Generate", variant="primary")

                with gr.Row():
                    clone_audio = gr.Audio(label="Output Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target, clone_language, clone_xvector, clone_size],
                    outputs=[clone_audio, clone_status],
                )

            # --- TTS (CustomVoice) ---
            with gr.Tab("TTS (CustomVoice)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize", lines=4,
                            value="Hello! Welcome to the Qwen3 text-to-speech system.",
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language", choices=LANGUAGES, value="English",
                            )
                            tts_speaker = gr.Dropdown(
                                label="Speaker", choices=SPEAKERS, value="Ryan",
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="Style Instruction (Optional)", lines=2,
                                placeholder="e.g. Speak in a cheerful tone",
                            )
                            tts_size = gr.Dropdown(
                                label="Model Size", choices=MODEL_SIZES, value="1.7B",
                            )
                        tts_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        tts_audio = gr.Audio(label="Output Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_size],
                    outputs=[tts_audio, tts_status],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
