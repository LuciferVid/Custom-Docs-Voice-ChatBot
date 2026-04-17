def retrieve_context(query: str, vector_store, top_k: int = 4, filter_doc: str = None) -> tuple[str, list[str]]:
    """
    Retrieves relevant chunks from the vector store and formats them as context.
    Returns (context_string, list_of_sources).
    """
    chunks = vector_store.search(query, top_k, filter_doc)
    
    if not chunks:
        return "No relevant context found.", []
        
    context_parts = []
    sources = []
    for chunk in chunks:
        source = chunk.get("source_file", "Unknown")
        page = chunk.get("page_number", "N/A")
        text = chunk.get("text", "")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{text}\n---")
        sources.append(f"{source} (Page {page})")
        
    return "\n".join(context_parts), list(set(sources))
