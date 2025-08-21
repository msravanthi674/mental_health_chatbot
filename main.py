import os
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel # You'll need to define ChatRequest in models.py
from fastapi.middleware.cors import CORSMiddleware

# Import your custom modules
from chat_engine import get_response
from crisis import contains_crisis_keywords, SAFETY_MESSAGE
from logger import log_chat
from doc_engine import query_documents

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow CORS for frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model (can also be in models.py)
class ChatRequest(BaseModel):
    session_id: str
    query: str

@app.get("/")
def read_root():
    """Welcome endpoint for the chatbot API."""
    return {"message": "Welcome to the AI-Powered Mental Health Chatbot!"}

@app.post("/chat")
async def chat_with_memory(request: ChatRequest):
    """
    Handles conversational chat with memory.
    Checks for crisis keywords and logs interactions.
    """
    session_id = request.session_id
    user_query = request.query

    # Crisis keyword check
    if contains_crisis_keywords(user_query):
        log_chat(session_id, user_query, SAFETY_MESSAGE, is_crisis=True)
        return {"response": SAFETY_MESSAGE}

    # Normal LLM response
    response = get_response(session_id, user_query)
    log_chat(session_id, user_query, response, is_crisis=False)
    return {"response": response}

@app.post("/doc-chat")
async def chat_with_documents(request: ChatRequest):
    """
    Handles queries against indexed documents.
    """
    response = query_documents(request.query)
    # Note: For doc-chat, you might want separate logging or
    # integrate crisis check if documents could contain sensitive info.
    # For simplicity, not adding full crisis/logging here for doc-chat,
    # but consider it for a production app.
    return {"response": response}

# To run this file:
# Save it as main.py
# Run `uvicorn main:app --reload` in your terminal