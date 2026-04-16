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
    Generates an embedding using Gemini's Cloud API.
    """
    client = get_client()
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        # Handle different response formats in SDK
        if hasattr(result.embeddings[0], 'values'):
            return result.embeddings[0].values
        return result.embeddings[0]
    except Exception as e:
        logger.error(f"Cloud Embedding Error: {e}")
        raise ValueError(f"Failed to generate embedding: {str(e)}")

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a batch using Gemini's Cloud API.
    """
    if not texts:
        return []
        
    client = get_client()
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=texts
        )
        # Robust parsing for list of Embedding objects
        embeddings = []
        for e in result.embeddings:
            if hasattr(e, 'values'):
                embeddings.append(e.values)
            else:
                embeddings.append(e)
        return embeddings
    except Exception as e:
        logger.error(f"Cloud Batch Embedding Error: {e}")
        raise ValueError(f"Failed to generate batch embeddings: {str(e)}")
