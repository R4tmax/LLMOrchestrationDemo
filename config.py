# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# set this to your local vault, link is designed as absolute, but it should work as relative as well should you chose to do so
# note that Win users should stick to absolute paths for their own mental healths sake
OBSIDIAN_VAULT_PATH = r"C:\Users\kadle\Documents\PersonalLibrary"

# Embedding model (local)
# this is what we use to create the vector embeddings
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Hardcoded paths for creating/retrieving data by the code
VECTOR_STORE_INDEX_PATH = "vector_store.faiss"
VECTOR_STORE_DOCS_PATH = "docs_store.pkl"

# I am using my own GCP cloud keys here, use your personal or choose a different model, refer to crewAI docs for viable interfaces
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- LEGACY ---

# Hugging Face Online Inference LLM
#HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#HF_TOKEN= os.getenv("HF-TOKEN")

#HF_MODEL_ID = "Qwen/Qwen3-235B-A22B"
#HF_MODEL_ID = "google/flan-t5-large"
#HF_TRANSFORMERS_MODEL_ID = "microsoft/phi-2" # Or another model

#HF_INFERENCE_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

