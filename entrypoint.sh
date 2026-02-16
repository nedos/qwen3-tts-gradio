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
echo "Starting Gradio app + OpenAI API..."
echo "================================"

# Activate venv
source /app/venv/bin/activate

# Start OpenAI API server in background
echo "Starting OpenAI API server on port 8002..."
python3 /app/openai_api.py &
API_PID=$!

# Start Gradio app
echo "Starting Gradio app on port 7860..."
exec python3 /app/app.py

# Cleanup on exit
trap "kill $API_PID 2>/dev/null; exit" INT TERM
