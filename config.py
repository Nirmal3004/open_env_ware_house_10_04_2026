import os


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = API_KEY
