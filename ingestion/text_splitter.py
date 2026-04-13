from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a text string using tiktoken.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Splits document pages into chunks using RecursiveCharacterTextSplitter.
    Preserves metadata and adds chunk index.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = []
    chunk_index = 0
    
    for page in pages:
        texts = text_splitter.split_text(page["text"])
        for text in texts:
            chunks.append({
                "text": text,
                "source_file": page["source_file"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index
            })
            chunk_index += 1
            
    return chunks
