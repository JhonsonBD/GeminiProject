from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

# Configure using environment variable (Render will provide this)
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=api_key)

app = FastAPI()
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

class Query(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    try:
        response = model.generate_content(query.prompt)
        if response.text:
            return {"reply": response.text.strip()}
        else:
            return {"error": "No response generated"}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint for Render
@app.get("/")
async def health_check():
    return {"status": "healthy"}
