import os
import pdfplumber
import pypdf
from docx import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> list[dict]:
    pages = []
    doc_name = os.path.basename(file_path)
    
    # Strategy 1: pdfplumber (High precision)
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append({
                        "text": page_text.strip(),
                        "page_number": i + 1,
                        "source_file": doc_name
                    })
    except Exception as e:
        logger.error(f"pdfplumber failed for {file_path}: {e}")

    # Strategy 2: pypdf (Failover)
    if not pages:
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pages.append({
                            "text": page_text.strip(),
                            "page_number": i + 1,
                            "source_file": doc_name
                        })
        except Exception as e:
            logger.error(f"pypdf failed for {file_path}: {e}")

    return pages

def load_document(file_path: str) -> list[dict]:
    """
    Detects file type and routes to the correct loader.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            return load_pdf(file_path)
        elif ext == ".docx":
            return load_docx(file_path)
        elif ext in [".txt", ".md"]:
            return load_txt(file_path)
        else:
            logger.warning(f"Unsupported file type attempt: {ext}")
            return []
    except Exception as e:
        logger.error(f"Failed to load context from {file_path}: {e}")
        return []

def get_supported_extensions() -> list[str]:
    """
    Returns list of supported extensions.
    """
    return [".pdf", ".docx", ".txt", ".md"]
