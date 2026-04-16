import os
import pdfplumber
from docx import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> list[dict]:
    """
    Extracts text from a PDF file using pdfplumber.
    Returns a list of dictionaries with text, page number, and source file.
    """
    pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) >= 10:
                    pages.append({
                        "text": text.strip(),
                        "page_number": i + 1,
                        "source_file": os.path.basename(file_path)
                    })
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
    return pages

def load_docx(file_path: str) -> list[dict]:
    """
    Extracts text from a Word document (.docx).
    Groups paragraphs into chunks of ~500 characters.
    """
    pages = []
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        
        content = "\n".join(full_text)
        # Split into chunks of ~500 chars
        chunk_size = 500
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if len(chunk.strip()) >= 50:
                pages.append({
                    "text": chunk.strip(),
                    "page_number": "N/A",
                    "source_file": os.path.basename(file_path)
                })
    except Exception as e:
        logger.error(f"Error loading DOCX {file_path}: {e}")
    return pages

def load_txt(file_path: str) -> list[dict]:
    """
    Reads text content from a .txt or .md file.
    Splits into chunks of 500 characters.
    """
    pages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunk_size = 500
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if len(chunk.strip()) >= 50:
                pages.append({
                    "text": chunk.strip(),
                    "page_number": "N/A",
                    "source_file": os.path.basename(file_path)
                })
    except Exception as e:
        logger.error(f"Error loading TXT {file_path}: {e}")
    return pages

def load_document(file_path: str) -> list[dict]:
    """
    Detects file type and routes to the correct loader.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in [".txt", ".md"]:
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def get_supported_extensions() -> list[str]:
    """
    Returns list of supported extensions.
    """
    return [".pdf", ".docx", ".txt", ".md"]
