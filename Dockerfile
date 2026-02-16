# Reuse c3-comfyui base (CUDA 12.9 + torch + flash-attn pre-built)
FROM ghcr.io/compute3ai/c3-comfyui:latest

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
