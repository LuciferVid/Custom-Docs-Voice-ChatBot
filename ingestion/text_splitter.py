def split_into_chunks(pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    """
    Splits document pages into chunks natively without LangChain.
    Preserves metadata and adds chunk index.
    """
    chunks = []
    chunk_index = 0
    
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            # Find the best breakpoint (newline or space) within the chunk
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Try to find a good breaking point like a newline or period
                break_point = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if break_point != -1 and break_point > start:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index
                })
                chunk_index += 1
            
            # Move start forward with overlap
            start = end - chunk_overlap if end < len(text) else len(text)
            
    return chunks
