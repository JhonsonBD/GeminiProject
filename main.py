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
    audio_data: str  # Base64 encoded audio
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
        print(f"🎵 Processing audio for session: {request.session_id}", flush=True)

        audio_bytes = base64.b64decode(request.audio_data)
        print(f"📥 Decoded audio: {len(audio_bytes)} bytes", flush=True)

        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)
        header = audio_buffer.read(12)
        audio_buffer.seek(0)

        if header.startswith(b'RIFF') and b'WAVE' in header:
            print("🎵 Processing WAV format", flush=True)
            y, sr = librosa.load(audio_buffer, sr=16000)
        else:
            print("🎵 Processing as raw PCM", flush=True)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            y = audio_np.astype(np.float32) / 32768.0
            if len(y) > 0:
                for orig_sr in [48000, 44100, 22050, 16000]:
                    try:
                        y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                        if len(y_resampled) > 0:
                            y = y_resampled
                            sr = 16000
                            break
                    except:
                        continue

        if len(y) == 0:
            raise HTTPException(status_code=400, detail="No audio data could be extracted")

        pcm_buffer = io.BytesIO()
        sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
        pcm_buffer.seek(0)
        pcm_audio = pcm_buffer.read()

        print(f"✅ Converted to PCM: {len(pcm_audio)} bytes", flush=True)

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
        print(f"💥 Unexpected error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_with_gemini_live(audio_data: bytes, config: dict) -> bytes:
    try:
        print(f"🔗 Connecting to Gemini Live model: {GEMINI_LIVE_MODEL}", flush=True)

        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:

            print(f"📤 Sending audio input: {len(audio_data)} bytes", flush=True)
            await session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )

            response_audio = io.BytesIO()

            wf = wave.open(response_audio, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)

            response_count = 0
            async for response in session.receive():
                if response.data is not None:
                    wf.writeframes(response.data)
                    response_count += 1
                    print(f"📥 Received response chunk {response_count}: {len(response.data)} bytes", flush=True)

            wf.close()
            response_audio.seek(0)

            final_audio = response_audio.read()
            print(f"✅ Final response audio: {len(final_audio)} bytes", flush=True)

            return final_audio

    except Exception as e:
        print(f"💥 Gemini Live processing error: {str(e)}", flush=True)
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

# Updated WebSocket endpoint with enhanced logging and error handling
@app.websocket("/live/stream/{session_id}")
async def websocket_live_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    try:
        while True:
            print(f"🔄 Waiting for message on WebSocket {session_id}", flush=True)
            data = await websocket.receive_text()
            print(f"📨 Raw message received: {data[:100]}...", flush=True)

            try:
                message = json.loads(data)
            except Exception as e:
                print(f"❌ Failed to parse JSON message: {e}", flush=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "session_id": session_id
                }))
                continue

            msg_type = message.get("type")
            print(f"📋 Parsed message type: {msg_type}", flush=True)

            if msg_type == "audio":
                try:
                    audio_data_b64 = message.get("audio")
                    if not audio_data_b64:
                        print("❌ No audio data found in message", flush=True)
                        continue

                    audio_data = base64.b64decode(audio_data_b64)
                    print(f"🔍 Decoded audio data size: {len(audio_data)} bytes", flush=True)

                    audio_buffer = io.BytesIO(audio_data)
                    audio_buffer.seek(0)
                    header = audio_buffer.read(12)
                    audio_buffer.seek(0)

                    if header.startswith(b'RIFF') and b'WAVE' in header:
                        print("🎵 Detected WAV format", flush=True)
                        y, sr = librosa.load(audio_buffer, sr=16000)
                    else:
                        print("❌ Unsupported audio format received, skipping chunk", flush=True)
                        # Optionally send error back:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Unsupported audio format, please send WAV audio",
                            "session_id": session_id
                        }))
                        continue

                    if len(y) == 0:
                        print("❌ No audio data extracted", flush=True)
                        continue

                    pcm_buffer = io.BytesIO()
                    sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
                    pcm_buffer.seek(0)
                    pcm_audio = pcm_buffer.read()

                    print(f"🔄 PCM audio size: {len(pcm_audio)} bytes", flush=True)

                    print("🤖 Sending to Gemini Live...", flush=True)
                    response_audio = await process_with_gemini_live(pcm_audio, active_sessions[session_id]["config"])

                    print(f"✅ Got response audio: {len(response_audio)} bytes", flush=True)

                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": base64.b64encode(response_audio).decode('utf-8'),
                        "session_id": session_id
                    }))

                    print("📤 Sent response to client", flush=True)

                except Exception as processing_error:
                    print(f"❌ Processing error: {processing_error}", flush=True)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Processing failed: {str(processing_error)}",
                        "session_id": session_id
                    }))

            else:
                print(f"⚠️ Unknown message type: {msg_type}", flush=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "session_id": session_id
                }))

    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected for session {session_id}", flush=True)
    except Exception as e:
        print(f"💥 WebSocket critical error: {e}", flush=True)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
