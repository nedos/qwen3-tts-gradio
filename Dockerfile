# Reuse c3-comfyui base (CUDA 12.9 + torch + flash-attn pre-built)
ARG COMFYUI_BASE_IMAGE=ghcr.io/compute3ai/c3-comfyui:latest
ARG MAX_JOBS=2
ARG EXT_PARALLEL=1
FROM ${COMFYUI_BASE_IMAGE}

# Re-declare build args after FROM and expose as env for downstream builds
ARG MAX_JOBS
ARG EXT_PARALLEL
ENV MAX_JOBS=${MAX_JOBS} \
    EXT_PARALLEL=${EXT_PARALLEL} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

WORKDIR /app

SHELL ["/bin/bash", "-c"]

# Install sox (required by qwen-tts for audio processing)
RUN apt-get update && apt-get install -y sox && rm -rf /var/lib/apt/lists/*

# Install qwen-tts + audio encoding deps (torch/flash-attn already in base)
RUN source /app/venv/bin/activate && \
    pip install --no-cache-dir \
    "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git" \
    huggingface_hub \
    pydub \
    soundfile

# Expose Gradio port
EXPOSE 7860

# Copy app and entrypoint (last for fast rebuilds)
COPY download.py /app/download.py
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
