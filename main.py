import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from chat import chat

app = FastAPI(title="مدرك")

history: list = []
 
 
class ChatRequest(BaseModel):
    message: str
 
 
class ChatResponse(BaseModel):
    reply: str
 
 
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    reply = chat(req.message, history)
    history.append(("user", req.message))
    history.append(("assistant", reply))
    return ChatResponse(reply=reply) 