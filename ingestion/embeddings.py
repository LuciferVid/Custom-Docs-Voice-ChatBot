import logging
import os
from google import genai

logger = logging.getLogger(__name__)

# No local model loading = Zero RAM usage
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)

def generate_embedding(text: str) -> list[float]:
    """
    Generates an embedding using Gemini's Cloud API (Free & Stable).
    """
    client = get_client()
    try:
        # text-embedding-004 is state-of-the-art and free
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.error(f"Cloud Embedding Error: {e}")
        return []

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a batch using Gemini's Cloud API.
    """
    client = get_client()
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=texts
        )
        return [e.values for e in result.embeddings]
    except Exception as e:
        logger.error(f"Cloud Batch Embedding Error: {e}")
        return []
