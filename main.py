# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

# קונפיגורציה
genai.configure(api_key="AIzaSyDB7hY-Otzl8l6q5x4aF8AOOwjNDYAlC7Q")

model = genai.GenerativeModel("gemini-pro")

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    try:
        response = model.generate_content(query.prompt)
        return {"reply": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}
