# huggingface_login.py (run once)
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(os.getenv("HF_TOKEN"))
