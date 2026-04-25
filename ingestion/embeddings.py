import logging
import os
from google import genai

import time
import random

logger = logging.getLogger(__name__)

# No local model loading = Zero RAM usage
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client

def generate_embedding_with_retry(contents, model="gemini-embedding-001", max_retries=5):
    """
    Helper to perform embedding with exponential backoff on 429s.
    """
    client = get_client()
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=model,
                contents=contents
            )
            return result
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str and attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16, 32 seconds + jitter
                sleep_time = (2 ** (attempt + 1)) + random.uniform(0, 1)
                logger.warning(f"Rate limited (429). Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                continue
            
            # If it's not a 429 or we're out of retries, log and re-raise
            logger.error(f"Embedding API Error (Attempt {attempt + 1}): {e}")
            raise e

def generate_embedding(text: str) -> list[float]:
    """
    Generates an embedding using Gemini's Cloud API with retries.
    """
    try:
        result = generate_embedding_with_retry(text)
        embedding = result.embeddings[0]
        return embedding.values if hasattr(embedding, 'values') else embedding
    except Exception as e:
        logger.error(f"Final Embedding Failure: {e}")
        raise ValueError(f"Failed to generate embedding after retries: {str(e)}")

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a batch using Gemini's Cloud API with retries.
    Handles Gemini's internal batch limits (max 100 per call).
    """
    if not texts:
        return []
        
    all_embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            result = generate_embedding_with_retry(batch)
            embeddings = [e.values if hasattr(e, 'values') else e for e in result.embeddings]
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Final Batch Embedding Failure (Batch {i//batch_size}): {e}")
            raise ValueError(f"Failed to generate batch embeddings after retries: {str(e)}")
                
    return all_embeddings
