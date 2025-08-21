# doc_engine.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.mistralai import MistralAI # Import MistralAI LLM for LlamaIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # Import for local embeddings
from dotenv import load_dotenv

load_dotenv() # Load environment variables

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is not set in the environment variables for doc_engine.")

# Initialize MistralAI LLM for LlamaIndex
mistral_llm = MistralAI(model="mistral-tiny", api_key=MISTRAL_API_KEY)

# Initialize a local embedding model
# This will download a model like 'BAAI/bge-small-en-v1.5' (default for HuggingFaceEmbedding)
# or you can specify another model from Hugging Face Model Hub.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# Load documents from the 'data' directory
try:
    documents = SimpleDirectoryReader("data").load_data()
    # Pass the local embedding model to VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=mistral_llm)
except Exception as e:
    print(f"Warning: Could not load documents or create LlamaIndex: {e}")
    query_engine = None

def query_documents(user_query: str) -> str:
    """
    Queries the indexed documents to find relevant information.
    """
    if query_engine is None:
        return "Document search functionality is currently unavailable."
    return str(query_engine.query(user_query))