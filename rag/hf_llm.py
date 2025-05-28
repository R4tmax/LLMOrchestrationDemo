# huggingface_login.py (run once)
from transformers import pipeline
from huggingface_hub import login,InferenceClient
import os
from dotenv import load_dotenv


#preload the HF token
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=os.getenv("HF_TOKEN"))

def query_hf(prompt):
    return client.text_generation(prompt, max_new_tokens=200)