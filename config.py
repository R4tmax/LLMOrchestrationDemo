# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OBSIDIAN_VAULT_PATH = r"C:\Users\kadle\Documents\PersonalLibrary"
VECTOR_STORE_INDEX_PATH = "vector_store.faiss"
VECTOR_STORE_DOCS_PATH = "docs_store.pkl"

# Hugging Face Online Inference LLM
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Choose a model available via Inference API, e.g., a Mixtral or Llama model
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" # Example
HF_INFERENCE_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Embedding model (local)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'