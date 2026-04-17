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
            model="gemini-embedding-001",
            contents=text
        )
        # Handle different response formats in SDK
        embedding = result.embeddings[0]
        return embedding.values if hasattr(embedding, 'values') else embedding
    except Exception as e:
        logger.error(f"Cloud Embedding Error: {e}")
        # Secondary fallback if text-embedding-004 is not available
        try:
             result = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
             embedding = result.embeddings[0]
             return embedding.values if hasattr(embedding, 'values') else embedding
        except Exception as e2:
             logger.error(f"Fallback Embedding Error: {e2}")
             raise ValueError(f"Failed to generate embedding: {str(e2)}")

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a batch using Gemini's Cloud API.
    """
    if not texts:
        return []
        
    client = get_client()
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        return [e.values if hasattr(e, 'values') else e for e in result.embeddings]
    except Exception as e:
        logger.error(f"Cloud Batch Embedding Error: {e}")
        # Secondary fallback
        try:
             result = client.models.embed_content(
                model="text-embedding-004",
                contents=texts
            )
             return [e.values if hasattr(e, 'values') else e for e in result.embeddings]
        except Exception as e2:
             logger.error(f"Fallback Batch Embedding Error: {e2}")
             raise ValueError(f"Failed to generate batch embeddings: {str(e2)}")
