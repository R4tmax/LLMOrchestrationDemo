# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OBSIDIAN_VAULT_PATH = r"C:\Users\kadle\Documents\PersonalLibrary"
VECTOR_STORE_INDEX_PATH = "vector_store.faiss"
VECTOR_STORE_DOCS_PATH = "docs_store.pkl"

# Hugging Face Online Inference LLM
HF_API_TOKEN = "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"
HF_TOKEN= "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"
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#HF_MODEL_ID = "google/flan-t5-large"
HF_TRANSFORMERS_MODEL_ID = "microsoft/phi-2" # Or another model

#HF_INFERENCE_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Embedding model (local)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'