# openai_api.py — OpenAI-compatible TTS API for Qwen3-TTS
# Add to your qwen3-tts-gradio repo

import os
import io
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Literal
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-TTS OpenAI API")

# Import from your existing app
from app import get_model, _normalize_audio

# Global model cache (loaded on startup)
loaded_models = {}

# Available speakers from your Gradio app
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
    "Serena", "Sohee", "Uncle_fu", "Vivian",
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
]


def get_tts_model(model_type: str = "CustomVoice", model_size: str = "1.7B"):
    """Lazy-load and cache TTS model."""
    key = (model_type, model_size)
    if key not in loaded_models:
        logger.info(f"Loading Qwen3-TTS {model_type} {model_size}...")
        loaded_models[key] = get_model(model_type, model_size)
        logger.info(f"Model {model_type}:{model_size} ready.")
    return loaded_models[key]


@app.on_event("startup")
async def startup_event():
    """Pre-load default model on startup."""
    logger.info("Pre-loading Qwen3-TTS models...")
    get_tts_model("CustomVoice", "1.7B")  # Default model
    logger.info("Startup complete.")


@app.get("/health")
async def health_check():
    """Health check for load balancers."""
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


def audio_to_bytes(audio_data: np.ndarray, sr: int, format: str = "mp3") -> io.BytesIO:
    """Convert audio numpy array to compressed bytes."""
    buf = io.BytesIO()
    
    if format == "wav":
        sf.write(buf, audio_data, sr, format='WAV')
        buf.seek(0)
        return buf, "audio/wav"
    
    # For compressed formats, write WAV first then convert
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_data, sr, format='WAV')
    wav_buf.seek(0)
    
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_buf)
        
        if format == "mp3":
            audio.export(buf, format='mp3', bitrate='192k')
            content_type = "audio/mpeg"
        elif format == "ogg":
            audio.export(buf, format='ogg', codec='libopus')
            content_type = "audio/ogg"
        elif format == "m4a":
            audio.export(buf, format='mp4', codec='aac')
            content_type = "audio/mp4"
        else:
            # Fallback to wav
            wav_buf.seek(0)
            return wav_buf, "audio/wav"
            
    except ImportError:
        # pydub not available, fallback to wav
        wav_buf.seek(0)
        return wav_buf, "audio/wav"
    
    buf.seek(0)
    return buf, content_type


# OpenAI-compatible TTS endpoint
@app.post("/v1/audio/speech")
async def create_speech(
    model: str = Form("tts-1"),
    input: str = Form(..., description="Text to synthesize"),
    voice: str = Form("Aiden", description="Speaker voice"),
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Form("mp3"),
    speed: float = Form(1.0, ge=0.25, le=4.0),
):
    """
    OpenAI-compatible text-to-speech endpoint.
    
    Maps to Qwen3-TTS CustomVoice model.
    """
    try:
        # Validate voice
        if voice not in SPEAKERS:
            voice = "Aiden"  # Fallback to default
        
        # Load model
        tts = get_tts_model("CustomVoice", "1.7B")
        
        # Generate speech
        logger.info(f"Generating speech for: {input[:50]}... voice={voice}")
        
        wavs, sr = tts.generate_speech(
            text=input,
            speaker=voice,
            language="Auto",  # Auto-detect
        )
        
        # Handle speed adjustment (if needed, Qwen3 may handle internally)
        audio_data = wavs[0] if isinstance(wavs, list) else wavs
        
        # Convert to compressed format
        buf, content_type = audio_to_bytes(audio_data, sr, response_format)
        
        return StreamingResponse(
            buf, 
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{response_format}"}
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional endpoints for voice cloning (non-OpenAI but useful)
@app.post("/v1/audio/voice_clone")
async def clone_voice(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    response_format: Literal["mp3", "ogg", "m4a", "wav"] = Form("mp3"),
    speed: float = Form(1.0),
):
    """
    Voice cloning endpoint using Qwen3-TTS Base model.
    
    - Accepts reference audio (wav, mp3, m4a, ogg)
    - Returns synthesized speech in cloned voice
    - Output format: mp3, ogg, m4a, or wav
    """
    try:
        import tempfile
        
        # Load base model for cloning
        tts = get_tts_model("Base", "1.7B")
        
        # Save uploaded reference audio temporarily
        ref_suffix = Path(reference_audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ref_suffix) as ref_tmp:
            content = await reference_audio.read()
            ref_tmp.write(content)
            ref_path = ref_tmp.name
        
        # Convert to wav if needed (Qwen3-TTS expects wav)
        if not ref_path.endswith('.wav'):
            wav_path = ref_path.rsplit('.', 1)[0] + '.wav'
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(ref_path)
                audio = audio.set_channels(1).set_frame_rate(24000)
                audio.export(wav_path, format='wav')
                ref_path = wav_path
            except Exception as e:
                logger.warning(f"Could not convert audio to wav: {e}, using original")
        
        # Transcribe reference audio (WhisperX or Qwen3-TTS may do this)
        # For now, use default reference text
        ref_text = "Reference audio sample."
        
        # Generate with cloned voice
        logger.info(f"Cloning voice for: {text[:50]}...")
        wavs, sr = tts.generate_speech(
            text=text,
            reference_audio=ref_path,
            reference_text=ref_text,
        )
        
        # Cleanup
        os.unlink(ref_path)
        if ref_path != ref_tmp.name and os.path.exists(ref_tmp.name):
            os.unlink(ref_tmp.name)
        
        # Return audio in requested format
        audio_data = wavs[0] if isinstance(wavs, list) else wavs
        buf, content_type = audio_to_bytes(audio_data, sr, response_format)
        
        return StreamingResponse(
            buf, 
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=cloned_speech.{response_format}"}
        )
        
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Voice design endpoint
@app.post("/v1/audio/voice_design")
async def design_voice(
    text: str = Form(...),
    voice_description: str = Form(...),
    language: str = Form("Auto"),
    response_format: Literal["mp3", "ogg", "m4a", "wav"] = Form("mp3"),
):
    """
    Voice design from natural language description.
    
    Uses Qwen3-TTS VoiceDesign model.
    
    - voice_description: e.g., "A deep, authoritative male voice with a British accent"
    - Returns synthesized speech with designed voice
    """
    try:
        tts = get_tts_model("VoiceDesign", "1.7B")
        
        logger.info(f"Designing voice: {voice_description[:50]}...")
        wavs, sr = tts.generate_voice_design(
            text=text,
            language=language,
            instruct=voice_description,
        )
        
        # Return audio in requested format
        audio_data = wavs[0] if isinstance(wavs, list) else wavs
        buf, content_type = audio_to_bytes(audio_data, sr, response_format)
        
        return StreamingResponse(
            buf, 
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=designed_voice.{response_format}"}
        )
        
    except Exception as e:
        logger.error(f"Voice design failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# List available voices
@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices (OpenAI-compatible)."""
    return {
        "voices": [
            {"voice_id": speaker, "name": speaker}
            for speaker in SPEAKERS
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
