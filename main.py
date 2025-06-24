from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import Request, Query, FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import io
import os
import uuid
import wave
import json
import base64
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, Optional
from google import genai
from google.genai import types
from pydub import AudioSegment

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Gemini Live Real-time API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: Dict[str, dict] = {}

GEMINI_LIVE_MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful AI assistant. Respond naturally and conversationally. Keep responses concise but friendly.",
}

class LiveSessionStart(BaseModel):
    user_id: str
    system_instruction: Optional[str] = None

class AudioMessage(BaseModel):
    session_id: str
    audio_data: str
    mime_type: str = "audio/wav"

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Gemini Live Real-time API"}

@app.post("/live/start")
async def start_live_session(request: LiveSessionStart):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/live/audio")
async def process_audio(request: AudioMessage):
    try:
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_info = active_sessions[request.session_id]
        audio_bytes = base64.b64decode(request.audio_data)
        audio_buffer = io.BytesIO(audio_bytes)

        audio_buffer.seek(0)
        header = audio_buffer.read(12)
        audio_buffer.seek(0)

        try:
            if header.startswith(b'RIFF') and b'WAVE' in header:
                y, sr = librosa.load(audio_buffer, sr=16000)
            else:
                temp_input = "/tmp/temp_input.webm"
                temp_output = "/tmp/temp_output.wav"
                with open(temp_input, "wb") as f:
                    f.write(audio_bytes)
                os.system(f"ffmpeg -y -i {temp_input} -ar 16000 -ac 1 {temp_output}")
                y, sr = librosa.load(temp_output, sr=16000)
                os.remove(temp_input)
                os.remove(temp_output)
        except Exception as decode_error:
            raise HTTPException(status_code=400, detail="Failed to decode audio")

        if len(y) == 0:
            raise HTTPException(status_code=400, detail="No audio data extracted")

        pcm_buffer = io.BytesIO()
        sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
        pcm_buffer.seek(0)
        pcm_audio = pcm_buffer.read()

        response_audio = await process_with_gemini_live(pcm_audio, session_info["config"])

        response_audio_b64 = base64.b64encode(response_audio).decode('utf-8')

        return {
            "session_id": request.session_id,
            "response_audio": response_audio_b64,
            "mime_type": "audio/wav",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.websocket("/live/stream/{session_id}")
async def websocket_live_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "session_id": session_id
                }))
                continue

            msg_type = message.get("type")
            if msg_type == "audio":
                try:
                    audio_data_b64 = message.get("audio")
                    if not audio_data_b64:
                        continue

                    audio_data = base64.b64decode(audio_data_b64)
                    audio_buffer = io.BytesIO(audio_data)
                    audio_buffer.seek(0)
                    header = audio_buffer.read(12)
                    audio_buffer.seek(0)

                    try:
                        if header.startswith(b'RIFF') and b'WAVE' in header:
                            y, sr = librosa.load(audio_buffer, sr=16000)
                        else:
                            temp_input = "/tmp/temp_input.webm"
                            temp_output = "/tmp/temp_output.wav"
                            with open(temp_input, "wb") as f:
                                f.write(audio_data)
                            os.system(f"ffmpeg -y -i {temp_input} -ar 16000 -ac 1 {temp_output}")
                            y, sr = librosa.load(temp_output, sr=16000)
                            os.remove(temp_input)
                            os.remove(temp_output)
                    except Exception as decode_error:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Audio decoding failed",
                            "session_id": session_id
                        }))
                        continue

                    if len(y) == 0:
                        continue

                    pcm_buffer = io.BytesIO()
                    sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
                    pcm_buffer.seek(0)
                    pcm_audio = pcm_buffer.read()

                    response_audio = await process_with_gemini_live(pcm_audio, active_sessions[session_id]["config"])

                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": base64.b64encode(response_audio).decode('utf-8'),
                        "session_id": session_id
                    }))

                except Exception as processing_error:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Processing failed: {str(processing_error)}",
                        "session_id": session_id
                    }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "session_id": session_id
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
