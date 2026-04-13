# Global model cache for lazy loading
_model = None

def get_model():
    """
    Lazy loads the SentenceTransformer model on first request.
    This prevents port-binding timeouts on platforms like Render.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def generate_embedding(text: str) -> list[float]:
    """
    Generates an embedding for a single text string.
    """
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a batch of text strings.
    """
    model = get_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()
