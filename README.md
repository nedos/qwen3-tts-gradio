# qwen3-tts-gradio

A Docker container for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with a multi-tab Gradio interface supporting voice design, voice cloning, and custom voice TTS.

## Quick Start

```bash
docker compose up
```

Then open http://localhost:7860. Models download on first use and are cached in a persistent volume.

## Pre-downloading Models

Set `PRELOAD_MODELS` to download weights at container startup instead of on first request:

```bash
PRELOAD_MODELS="CustomVoice:1.7B" docker compose up
```

Multiple models can be comma-separated:

```bash
PRELOAD_MODELS="CustomVoice:1.7B,Base:1.7B,VoiceDesign:1.7B" docker compose up
```

### Available Models

| Type | Sizes | Description |
|------|-------|-------------|
| `CustomVoice` | 0.6B, 1.7B | Predefined speakers with optional style instructions |
| `VoiceDesign` | 1.7B | Create voices from natural language descriptions |
| `Base` | 0.6B, 1.7B | Clone a voice from a reference audio clip |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRELOAD_MODELS` | Models to download at startup (e.g. `CustomVoice:1.7B,Base:0.6B`) | - |
| `HF_TOKEN` | HuggingFace token (if needed for gated models) | - |

## Building

```bash
git clone --recurse-submodules https://github.com/nedos/qwen3-tts-gradio
cd qwen3-tts-gradio
docker build -t qwen3-tts-gradio .
```

## Credits

- 🎙️ [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team

## License

MIT
