import os


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
OPENAI_API_KEY = API_KEY
