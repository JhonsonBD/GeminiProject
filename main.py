from fastapi.responses import JSONResponse
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
import subprocess

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
            total_bytes = 0
            async for response in session.receive():
                if response.data:
                    wf.writeframes(response.data)
                    total_bytes += len(response.data)
                    response_count += 1
                    print(f"📥 Received chunk {response_count}: {len(response.data)} bytes", flush=True)

            wf.close()
            response_audio.seek(0)

            final_audio = response_audio.read()
            print(f"✅ Final response audio: {len(final_audio)} bytes", flush=True)

            return final_audio

    except Exception as e:
        print(f"💥 Gemini Live processing error: {str(e)}", flush=True)
        raise Exception(f"Gemini Live processing error: {str(e)}")

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

            if message.get("type") != "audio":
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Unsupported message type",
                    "session_id": session_id
                }))
                continue

            try:
                audio_data = base64.b64decode(message["audio"])
                input_path = "/tmp/temp_input.webm"
                output_path = "/tmp/temp_output.wav"

                with open(input_path, "wb") as f:
                    f.write(audio_data)

                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-ar", "16000", "-ac", "1",
                    "-f", "wav", output_path
                ]
                subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                with open(output_path, "rb") as f:
                    wav_data = f.read()

                y, sr = librosa.load(io.BytesIO(wav_data), sr=16000)
                pcm_buffer = io.BytesIO()
                sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
                pcm_buffer.seek(0)

                response_audio = await process_with_gemini_live(pcm_buffer.read(), active_sessions[session_id]["config"])

                await websocket.send_text(json.dumps({
                    "type": "audio_response",
                    "audio": base64.b64encode(response_audio).decode("utf-8"),
                    "session_id": session_id
                }))

            except Exception as e:
                print(f"❌ Error processing audio: {e}", flush=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Failed to process audio: {str(e)}",
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
