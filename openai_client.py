from openai import OpenAI

from config import API_BASE_URL, API_KEY


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
