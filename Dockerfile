# Qwen3-TTS Gradio — follows c3-comfyui layer structure for cache sharing
# Layers up to flash-attn are identical to c3-comfyui
ARG MAX_JOBS=2
ARG EXT_PARALLEL=1

# Same base as c3-comfyui
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG MAX_JOBS
ARG EXT_PARALLEL

# Same CUDA arch list as c3-comfyui
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

ENV PYTHONUNBUFFERED=1 \
    MAX_JOBS=${MAX_JOBS} \
    EXT_PARALLEL=${EXT_PARALLEL} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Same apt layer as c3-comfyui (python, git, ffmpeg)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /app/venv

SHELL ["/bin/bash", "-c"]

# Torch — pinned to 2.6.0+cu126 (torch 2.10 causes CUBLAS_STATUS_INVALID_VALUE on L4/Ada bf16 gemm)
RUN source /app/venv/bin/activate && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# flash-attn — pinned to 2.7.4.post1 (source build, no prebuilt wheels for flash-attn 2.x)
RUN source /app/venv/bin/activate && \
    pip install packaging ninja wheel psutil && \
    pip install flash-attn==2.7.4.post1 --no-build-isolation

# === TTS-specific layers below (diverges from c3-comfyui here) ===

# Install sox (required by qwen-tts for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends sox && rm -rf /var/lib/apt/lists/*

# Install qwen-tts + audio encoding deps
RUN source /app/venv/bin/activate && \
    pip install --no-cache-dir \
    "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git" \
    huggingface_hub \
    pydub \
    soundfile

EXPOSE 7860

# Copy app files (last for fast rebuilds)
COPY download.py /app/download.py
COPY app.py /app/app.py
COPY openai_api.py /app/openai_api.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
