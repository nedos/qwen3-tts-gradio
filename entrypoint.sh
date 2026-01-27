#!/bin/bash
set -e

echo "Qwen3-TTS Container Starting..."
echo "================================"

# Handle empty HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    unset HF_TOKEN
fi

# Pre-download models if requested
if [ -n "${PRELOAD_MODELS}" ]; then
    echo "Running model downloader..."
    source /app/venv/bin/activate && python3 /app/download.py
fi

echo ""
echo "Starting Gradio app..."
echo "================================"

exec /bin/bash -c "source /app/venv/bin/activate && python3 /app/app.py"
