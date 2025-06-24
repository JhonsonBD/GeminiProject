from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv

import requests
import asyncio
import uuid
import base64
import json
import io
import os
import wave

import numpy as np
import soundfile as sf
import librosa
from google import genai
from google.genai import types

# Load .env
load_dotenv()

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

client = genai.Client(api_key=api_key)

# FastAPI app setup
app = FastAPI(title="Gemini Live Real-time API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active session store
active_sessions: Dict[str, dict] = {}

# Gemini Live configuration
GEMINI_LIVE_MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful AI assistant. Respond naturally and conversationally. Keep responses concise but friendly.",
}

# === Models ===

class LiveSessionStart(BaseModel):
    user_id: str
    system_instruction: Optional[str] = None

class AudioMessage(BaseModel):
    session_id: str
    audio_data: str  # Base64 encoded
    mime_type: str = "audio/wav"

# === Routes ===

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Gemini Live Real-time API"}

@app.post("/live/start")
async def start_live_session(request: LiveSessionStart):
    session_id = str(uuid.uuid4())

    config = LIVE_CONFIG.copy()
    if request.system_instruction:
        config["system_instruction"] = request.system_instruction

    active_sessions[session_id] = {
        "user_id": request.user_id,
        "config": config,
        "status": "ready"
    }

    return {
        "session_id": session_id,
        "status": "ready",
        "message": "Gemini Live session started - ready for audio!"
    }

@app.post("/live/audio")
async def process_audio(request: AudioMessage):
    try:
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        audio_bytes = base64.b64decode(request.audio_data)
        audio_buffer = io.BytesIO(audio_bytes)

        audio_buffer.seek(0)
        header = audio_buffer.read(12)
        audio_buffer.seek(0)

        if header.startswith(b'RIFF') and b'WAVE' in header:
            print("🎵 Detected WAV format")
            y, sr = librosa.load(audio_buffer, sr=16000)
        else:
            print("🎵 Trying raw PCM format")
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            y = audio_np.astype(np.float32) / 32768.0
            y = librosa.resample(y, orig_sr=48000, target_sr=16000)
            sr = 16000

        if len(y) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data")

        pcm_buffer = io.BytesIO()
        sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
        pcm_buffer.seek(0)
        pcm_audio = pcm_buffer.read()

        response_audio = await process_with_gemini_live(pcm_audio, active_sessions[request.session_id]["config"])

        return {
            "session_id": request.session_id,
            "response_audio": base64.b64encode(response_audio).decode('utf-8'),
            "mime_type": "audio/wav",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/live/{session_id}")
async def end_live_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del active_sessions[session_id]
    return {"status": "session_ended", "session_id": session_id}

@app.get("/live/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "status": active_sessions[session_id]["status"],
        "user_id": active_sessions[session_id]["user_id"]
    }

# === WebSocket ===

@app.websocket("/live/stream/{session_id}")
async def websocket_live_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "audio":
                try:
                    audio_data = base64.b64decode(message["audio"])
                    audio_buffer = io.BytesIO(audio_data)
                    audio_buffer.seek(0)
                    header = audio_buffer.read(12)
                    audio_buffer.seek(0)

                    if header.startswith(b'RIFF') and b'WAVE' in header:
                        y, sr = librosa.load(audio_buffer, sr=16000)
                    else:
                        audio_np = np.frombuffer(audio_data, dtype=np.int16)
                        y = audio_np.astype(np.float32) / 32768.0
                        y = librosa.resample(y, orig_sr=48000, target_sr=16000)
                        sr = 16000

                    if y is None or len(y) == 0:
                        continue

                    pcm_buffer = io.BytesIO()
                    sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
                    pcm_buffer.seek(0)
                    pcm_audio = pcm_buffer.read()

                    if len(pcm_audio) == 0:
                        continue

                    response_audio = await process_with_gemini_live(
                        pcm_audio,
                        active_sessions[session_id]["config"]
                    )

                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": base64.b64encode(response_audio).decode('utf-8'),
                        "session_id": session_id
                    }))

                except Exception as err:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(err),
                        "session_id": session_id
                    }))

    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"💥 WebSocket critical error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal error")
        except:
            pass

# === Gemini Handler ===

async def process_with_gemini_live(audio_data: bytes, config: dict) -> bytes:
    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            await session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )

            response_audio = io.BytesIO()
            wf = wave.open(response_audio, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)

            async for response in session.receive():
                if response.data:
                    wf.writeframes(response.data)

            wf.close()
            response_audio.seek(0)
            return response_audio.read()

    except Exception as e:
        raise Exception(f"Gemini Live processing error: {str(e)}")
