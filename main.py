from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import Request
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import io
import os
from typing import Dict, Optional
import uuid
import wave
import json
from google import genai
from google.genai import types
import soundfile as sf
import librosa
import base64

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Gemini Live Real-time API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_sessions: Dict[str, dict] = {}

# Gemini Live configuration
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
    """Start a new Gemini Live session"""
    try:
        session_id = str(uuid.uuid4())
        
        # Custom system instruction if provided
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
    """Process audio through Gemini Live and return audio response"""
    try:
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = active_sessions[request.session_id]
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.audio_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Convert audio to required format (16kHz PCM)
        audio_buffer = io.BytesIO(audio_bytes)
        try:
            # Load audio and convert to 16kHz
            y, sr = librosa.load(audio_buffer, sr=16000)
            
            # Convert to PCM format
            pcm_buffer = io.BytesIO()
            sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
            pcm_buffer.seek(0)
            pcm_audio = pcm_buffer.read()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")
        
        # Process with Gemini Live
        response_audio = await process_with_gemini_live(pcm_audio, session_info["config"])
        
        # Convert response audio to base64
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
    """Process audio through Gemini Live API"""
    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            
            # Send audio input
            await session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )
            
            # Collect response audio
            response_audio = io.BytesIO()
            
            # Set up wave file for proper audio format
            wf = wave.open(response_audio, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)  # Gemini outputs 24kHz
            
            async for response in session.receive():
                if response.data is not None:
                    wf.writeframes(response.data)
            
            wf.close()
            response_audio.seek(0)
            
            return response_audio.read()
            
    except Exception as e:
        raise Exception(f"Gemini Live processing error: {str(e)}")

@app.delete("/live/{session_id}")
async def end_live_session(session_id: str):
    """End a live session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"status": "session_ended", "session_id": session_id}

@app.get("/live/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": active_sessions[session_id]["status"],
        "user_id": active_sessions[session_id]["user_id"]
    }

# WebSocket endpoint for real-time streaming (optional advanced feature)
@app.websocket("/live/stream/{session_id}")
async def websocket_live_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio":
                # Process audio through Gemini Live
                audio_data = base64.b64decode(message["audio"])
                
                # Convert to PCM format
                audio_buffer = io.BytesIO(audio_data)
                y, sr = librosa.load(audio_buffer, sr=16000)
                pcm_buffer = io.BytesIO()
                sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
                pcm_buffer.seek(0)
                pcm_audio = pcm_buffer.read()
                
                # Get response from Gemini Live
                response_audio = await process_with_gemini_live(
                    pcm_audio, 
                    active_sessions[session_id]["config"]
                )
                
                # Send response back
                await websocket.send_text(json.dumps({
                    "type": "audio_response",
                    "audio": base64.b64encode(response_audio).decode('utf-8'),
                    "session_id": session_id
                }))
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")

load_dotenv()

ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_REDIRECT_URI = os.getenv("ZOOM_REDIRECT_URI")

@app.get("/zoom/auth")
async def zoom_auth():
    return RedirectResponse(
        f"https://zoom.us/oauth/authorize?response_type=code&client_id={ZOOM_CLIENT_ID}&redirect_uri={ZOOM_REDIRECT_URI}"
    )

@app.get("/zoom/callback")
async def zoom_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "Missing code"}, status_code=400)

    token_url = "https://zoom.us/oauth/token"
    basic_auth = requests.auth._basic_auth_str(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET)
    headers = {
        "Authorization": basic_auth,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": ZOOM_REDIRECT_URI,
    }

    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code != 200:
        return JSONResponse({"error": "Failed to retrieve token", "details": response.text}, status_code=500)

    return JSONResponse({"message": "Success", "tokens": response.json()})

    return JSONResponse({"message": "Success", "tokens": tokens})


