# chat_engine.py
import os
from dotenv import load_dotenv # type: ignore
from langchain_mistralai import ChatMistralAI # type: ignore
from langchain.chains import ConversationChain # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore

# load .env
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is not set in the environment variables.")

# Initialize MistralAI LLM
# You can specify the model, e.g., "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"
llm = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, temperature=0.7, model="mistral-tiny")

# Store per-user memory sessions
session_memory_map = {}

def get_response(session_id: str, query: str) -> str:
    """
    Retrieves a conversational response for a given session ID and query.
    Maintains conversation memory for each session.
    """
    if session_id not in session_memory_map:
        memory = ConversationBufferMemory()
        session_memory_map[session_id] = ConversationChain(llm=llm, memory=memory, verbose=True)

    conversation = session_memory_map[session_id]
    return conversation.predict(input=query)