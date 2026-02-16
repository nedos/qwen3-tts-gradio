# Base image with CUDA runtime
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set CUDA architectures for building without GPUs
# 8.0=A100, 8.6=RTX30xx, 8.9=RTX40xx/L40S, 9.0=H100, 12.0=Blackwell
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/venv

# Use shell form for commands that need to source the activation script
SHELL ["/bin/bash", "-c"]

# Install torch first (stable layer - cached separately)
# cu128 is backward compatible with CUDA 12.9 runtime
RUN source /app/venv/bin/activate && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install flash-attn (separate layer - long compile time)
RUN source /app/venv/bin/activate && \
    pip install packaging ninja wheel psutil && \
    pip install flash-attn --no-build-isolation

# === Qwen3-TTS-specific layers below ===

# Install qwen-tts + audio encoding deps
RUN source /app/venv/bin/activate && \
    pip install --no-cache-dir \
    "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git" \
    huggingface_hub \
    pydub \
    soundfile

# Expose Gradio port
EXPOSE 7860

# Copy app and entrypoint (last to enable fast rebuilds)
COPY download.py /app/download.py
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
