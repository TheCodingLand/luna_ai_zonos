from contextlib import asynccontextmanager
import logging
import multiprocessing
import os
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from io import BytesIO
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from fastapi.responses import FileResponse, StreamingResponse
import time

multiprocessing.set_start_method("spawn", force=True)

# Directory to store voice cache
VOICE_CACHE_DIR = "voice_cache"
os.makedirs(VOICE_CACHE_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models and voice cache
    load_models()
    load_voice_cache()  # Load persisted voice cache
    yield
    # Unload the ML models and optionally save voice cache
    save_voice_cache()  # Save voice cache before shutdown
    MODELS["transformer"] = None
    MODELS["hybrid"] = None


app = FastAPI(title="Zonos API", description="OpenAI-compatible TTS API for Zonos", lifespan=lifespan)

# Model Management
MODELS: Dict[str, Zonos] = {"transformer": None, "hybrid": None}  # type: ignore
VOICE_CACHE: Dict[str, torch.Tensor] = {}


def load_models():
    """Load both models at startup and keep them in VRAM"""
    device = "cuda"
    logging.info("Loading models...")
    logging.info("Loading transformer model...")
    MODELS["transformer"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    MODELS["transformer"].requires_grad_(False).eval()
    logging.info("Loaded transformer model")
    logging.info("Loading hybrid model...")
    MODELS["hybrid"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    MODELS["hybrid"].requires_grad_(False).eval()
    logging.info("Loaded hybrid model")


def save_voice_cache():
    """Save all voice embeddings in VOICE_CACHE to disk."""
    logging.info("Saving voice cache to disk...")
    for voice_id, embedding in VOICE_CACHE.items():
        file_path = os.path.join(VOICE_CACHE_DIR, f"{voice_id}.pt")
        torch.save(embedding.cpu(), file_path)  # Save to CPU to avoid GPU memory issues
    logging.info(f"Saved {len(VOICE_CACHE)} voice embeddings.")


def load_voice_cache():
    """Load voice embeddings from disk into VOICE_CACHE."""
    logging.info("Loading voice cache from disk...")
    device = "cuda"
    for file_name in os.listdir(VOICE_CACHE_DIR):
        if file_name.endswith(".pt"):
            voice_id = file_name[:-3]  # Remove '.pt' extension
            file_path = os.path.join(VOICE_CACHE_DIR, file_name)
            embedding = torch.load(file_path, map_location=device)  # Load directly to GPU
            VOICE_CACHE[voice_id] = embedding
    logging.info(f"Loaded {len(VOICE_CACHE)} voice embeddings.")


# API Models (unchanged)
class SpeechRequest(BaseModel):
    model: str = Field("Zyphra/Zonos-v0.1-transformer", description="Model to use")
    input: str = Field(..., max_length=500, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice ID to use")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed multiplier")
    language: str = Field("en-us", description="Language code")
    emotion: Optional[Dict[str, float]] = None
    response_format: str = Field("mp3", description="Audio format (mp3 or wav)")


class VoiceResponse(BaseModel):
    voice_id: str
    created: int  # Unix timestamp


# API Endpoints
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    try:
        model: Zonos = MODELS["transformer" if "transformer" in request.model else "hybrid"]

        # Convert speed to speaking_rate (15.0 is default)
        speaking_rate = 15.0 * request.speed

        # Prepare emotion tensor if provided
        emotion_tensor = None
        if request.emotion:
            emotion_values = [
                request.emotion.get("happiness", 1.0),
                request.emotion.get("sadness", 0.05),
                request.emotion.get("disgust", 0.05),
                request.emotion.get("fear", 0.05),
                request.emotion.get("surprise", 0.05),
                request.emotion.get("anger", 0.05),
                request.emotion.get("other", 0.1),
                request.emotion.get("neutral", 0.2),
            ]
            emotion_tensor = torch.tensor(emotion_values, device="cuda").unsqueeze(0)

        # Get voice embedding from cache if provided
        speaker_embedding = VOICE_CACHE.get(request.voice) if request.voice else None

        # Default conditioning parameters
        cond_dict = make_cond_dict(
            text=request.input,
            language=request.language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            speaking_rate=speaking_rate,
            device="cuda",
            unconditional_keys=[] if request.emotion else ["emotion"],
        )

        conditioning = model.prepare_conditioning(cond_dict)

        # Generate audio
        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=2.0,
            batch_size=1,
            sampling_params=dict(min_p=0.15),
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        # Ensure proper shape
        if wav_out.dim() > 2:
            wav_out = wav_out.squeeze()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)

        # Convert to requested format
        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format=request.response_format)
        buffer.seek(0)

        # Determine the filename based on the response format
        filename = f"speech.{request.response_format}"

        # Streaming response for real-time audio
        return StreamingResponse(
            buffer,
            media_type=f"audio/{request.response_format}",
            headers={"Content-Disposition": f"inline; filename={filename}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/voice")
async def create_voice(file: UploadFile = File(...), name: str = None):
    try:
        # Read the audio file
        content = await file.read()
        audio_data = BytesIO(content)

        # Load and process audio
        wav, sr = torchaudio.load(audio_data)

        # Generate embedding using transformer model
        speaker_embedding: Zonos = MODELS["transformer"].make_speaker_embedding(wav, sr)

        # Generate unique voice ID and cache embedding
        timestamp = int(time.time())
        voice_id = f"{name}_voice_{timestamp}_{len(VOICE_CACHE)}"
        VOICE_CACHE[voice_id] = speaker_embedding.to("cuda")  # Ensure it's on GPU

        # Save the new embedding to disk immediately
        file_path = os.path.join(VOICE_CACHE_DIR, f"{voice_id}.pt")
        torch.save(speaker_embedding.cpu(), file_path)  # Save to CPU to avoid GPU memory issues

        return VoiceResponse(voice_id=voice_id, created=timestamp)

    except Exception as e:
        error_msg = str(e)
        if "cuda" in error_msg.lower():
            error_msg = "GPU error while processing voice. Please try again."
        elif "load" in error_msg.lower():
            error_msg = "Failed to load audio file. Please ensure it's a valid audio format."
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/v1/audio/voices")
async def list_voices():
    return list(VOICE_CACHE.keys())


class ZonosModel(BaseModel):
    id: str
    created: int
    object: str
    owned_by: str


class SupportedModelsResponse(BaseModel):
    models: List[ZonosModel]


transformer_model = ZonosModel(
    id="Zyphra/Zonos-v0.1-transformer", created=1234567890, object="model", owned_by="zyphra"
)
hybrid_model = ZonosModel(id="Zyphra/Zonos-v0.1-hybrid", created=1234567890, object="model", owned_by="zyphra")


@app.get("/v1/audio/models")
async def list_models() -> SupportedModelsResponse:
    """List available models and their status"""
    return SupportedModelsResponse(models=[transformer_model, hybrid_model])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
