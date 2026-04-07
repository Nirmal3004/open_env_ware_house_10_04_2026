import os


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
