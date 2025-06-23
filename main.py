from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List, Optional
import uuid
from datetime import datetime

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=api_key)

app = FastAPI(title="Real-time Gemini Chat API")

# Add CORS middleware for Bubble.io
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Bubble app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model with conversation-optimized settings
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro-latest",
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        top_p=0.8,
        top_k=40,
        max_output_tokens=1000,
    )
)

# In-memory conversation storage (use Redis/DB for production)
conversations = {}

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class ConversationStart(BaseModel):
    user_id: str
    system_prompt: Optional[str] = "אתה עוזר בינה מלאכותית שמתמחה בכל מני נושאים"

class ChatQuery(BaseModel):
    conversation_id: str
    user_message: str
    user_id: str

class ConversationResponse(BaseModel):
    conversation_id: str
    ai_response: str
    conversation_history: List[dict]
    status: str

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Real-time Gemini Chat API"}

@app.post("/conversation/start")
async def start_conversation(request: ConversationStart):
    """Start a new conversation session"""
    try:
        conversation_id = str(uuid.uuid4())
        
        conversations[conversation_id] = {
            "user_id": request.user_id,
            "messages": [],
            "system_prompt": request.system_prompt,
            "created_at": datetime.now()
        }
        
        return {
            "conversation_id": conversation_id,
            "status": "conversation_started",
            "message": "Ready for real-time chat!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/realtime", response_model=ConversationResponse)
async def realtime_chat(query: ChatQuery):
    """Handle real-time chat messages"""
    try:
        if query.conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation = conversations[query.conversation_id]
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": query.user_message,
            "timestamp": datetime.now().isoformat()
        }
        conversation["messages"].append(user_message)
        
        # Build conversation context
        context = conversation["system_prompt"] + "\n\n"
        for msg in conversation["messages"][-10:]:  # Keep last 10 messages for context
            context += f"{msg['role']}: {msg['content']}\n"
        
        # Generate AI response
        response = model.generate_content(context + "assistant:")
        
        if not response.text:
            ai_response = "לא הבנתי כל כך את מה שאמרת, תוכל לחזור על זה?"
        else:
            ai_response = response.text.strip()
        
        # Add AI response to conversation
        ai_message = {
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        conversation["messages"].append(ai_message)
        
        return ConversationResponse(
            conversation_id=query.conversation_id,
            ai_response=ai_response,
            conversation_history=conversation["messages"][-5:],  # Return last 5 messages
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get full conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]["messages"],
        "total_messages": len(conversations[conversation_id]["messages"])
    }

@app.delete("/conversation/{conversation_id}")
async def end_conversation(conversation_id: str):
    """End and cleanup conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"status": "conversation_ended", "conversation_id": conversation_id}

@app.post("/chat/quick")
async def quick_chat(query: dict):
    """Quick chat without conversation context (for simple use cases)"""
    try:
        user_prompt = query.get("prompt", "")
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        response = model.generate_content(user_prompt)
        
        return {
            "reply": response.text.strip() if response.text else "No response generated",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}
