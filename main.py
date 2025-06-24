from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import Request
from fastapi import Query
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
import numpy as np

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
        print(f"🎵 Processing audio for session: {request.session_id}")
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.audio_data)
            print(f"📥 Decoded audio: {len(audio_bytes)} bytes")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Enhanced audio processing
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Check if it's a valid audio format
            audio_buffer.seek(0)
            header = audio_buffer.read(12)
            audio_buffer.seek(0)
            
            if header.startswith(b'RIFF') and b'WAVE' in header:
                print("🎵 Processing WAV format")
                y, sr = librosa.load(audio_buffer, sr=16000)
            else:
                print("🎵 Processing as raw PCM")
                # Try as raw 16-bit PCM
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                y = audio_np.astype(np.float32) / 32768.0
                # Assume common sample rates and resample to 16kHz
                if len(y) > 0:
                    # Try different common sample rates
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
            
            # Convert to PCM format
            pcm_buffer = io.BytesIO()
            sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
            pcm_buffer.seek(0)
            pcm_audio = pcm_buffer.read()
            
            print(f"✅ Converted to PCM: {len(pcm_audio)} bytes")
            
        except Exception as e:
            print(f"❌ Audio processing error: {str(e)}")
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
        print(f"💥 Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_with_gemini_live(audio_data: bytes, config: dict) -> bytes:
    """Process audio through Gemini Live API with enhanced error handling"""
    try:
        print(f"🔗 Connecting to Gemini Live model: {GEMINI_LIVE_MODEL}")
        
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            
            print(f"📤 Sending audio input: {len(audio_data)} bytes")
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
            
            response_count = 0
            async for response in session.receive():
                if response.data is not None:
                    wf.writeframes(response.data)
                    response_count += 1
                    print(f"📥 Received response chunk {response_count}: {len(response.data)} bytes")
            
            wf.close()
            response_audio.seek(0)
            
            final_audio = response_audio.read()
            print(f"✅ Final response audio: {len(final_audio)} bytes")
            
            return final_audio
            
    except Exception as e:
        print(f"💥 Gemini Live processing error: {str(e)}")
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

# WebSocket endpoint for real-time streaming (fixed version)
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
            print(f"🔄 Waiting for message on WebSocket {session_id}")
            data = await websocket.receive_text()
            print(f"📨 Raw message received: {data[:100]}...")  # Show first 100 chars
            message = json.loads(data)
            print(f"📋 Parsed message type: {message.get('type')}")
            
            if message.get("type") == "audio":
                try:
                    print(f"📥 Received audio chunk for session {session_id}")
                    
                    # Process audio through Gemini Live
                    audio_data = base64.b64decode(message["audio"])
                    print(f"🔍 Decoded audio data size: {len(audio_data)} bytes")
                    
                    # Enhanced audio processing with better error handling
                    try:
                        # Method 1: Try direct librosa load with file format specification
                        audio_buffer = io.BytesIO(audio_data)
                        
                        # Try different approaches for audio loading
                        y = None
                        sr = None
                        
                        # First, try to detect if it's a WAV file
                        audio_buffer.seek(0)
                        header = audio_buffer.read(12)
                        audio_buffer.seek(0)
                        
                        if header.startswith(b'RIFF') and b'WAVE' in header:
                            print("🎵 Detected WAV format")
                            y, sr = librosa.load(audio_buffer, sr=16000)
                        else:
                            print("🎵 Trying raw audio format")
                            # Try as raw audio (common for WebRTC)
                            # Assume it's 16-bit PCM at 48kHz (common WebRTC format)
                            audio_np = np.frombuffer(audio_data, dtype=np.int16)
                            # Convert to float and normalize
                            y = audio_np.astype(np.float32) / 32768.0
                            # Resample to 16kHz if needed
                            if len(y) > 0:
                                y = librosa.resample(y, orig_sr=48000, target_sr=16000)
                                sr = 16000
                        
                        if y is None or len(y) == 0:
                            print("❌ No audio data extracted")
                            continue
                            
                        print(f"🎵 Processed audio: {len(y)} samples at {sr}Hz")
                        
                        # Convert to PCM format for Gemini
                        pcm_buffer = io.BytesIO()
                        sf.write(pcm_buffer, y, 16000, format='RAW', subtype='PCM_16')
                        pcm_buffer.seek(0)
                        pcm_audio = pcm_buffer.read()
                        
                        print(f"🔄 PCM audio size: {len(pcm_audio)} bytes")
                        
                    except Exception as audio_error:
                        print(f"❌ Audio processing error: {audio_error}")
                        # Send error response back to client
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Audio processing failed: {str(audio_error)}",
                            "session_id": session_id
                        }))
                        continue
                    
                    # Only proceed if we have valid PCM audio
                    if len(pcm_audio) == 0:
                        print("❌ Empty PCM audio, skipping")
                        continue
                    
                    # Get response from Gemini Live
                    print("🤖 Sending to Gemini Live...")
                    response_audio = await process_with_gemini_live(
                        pcm_audio, 
                        active_sessions[session_id]["config"]
                    )
                    
                    print(f"✅ Got response audio: {len(response_audio)} bytes")
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": base64.b64encode(response_audio).decode('utf-8'),
                        "session_id": session_id
                    }))
                    
                    print("📤 Sent response to client")
                    
                except Exception as processing_error:
                    print(f"❌ Processing error: {processing_error}")
                    # Send error but don't break the connection
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Processing failed: {str(processing_error)}",
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
