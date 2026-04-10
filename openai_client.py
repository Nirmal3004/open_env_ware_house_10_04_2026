from openai import OpenAI

from config import API_BASE_URL, OPENAI_API_KEY


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL,
    )
