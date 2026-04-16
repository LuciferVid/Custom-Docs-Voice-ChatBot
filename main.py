import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
from google import genai
from dotenv import load_dotenv

from ingestion.document_loader import load_document
from ingestion.text_splitter import split_into_chunks
from vector_store.faiss_store import FAISSVectorStore
from rag.memory import ConversationMemory
from rag.chain import get_answer
from voice.speech_to_text import transcribe_audio
from voice.text_to_speech import synthesize_speech

# Load environment variables
load_dotenv()

app = FastAPI(title="Voice RAG Chatbot API")

# Global client cache
_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

vector_store = FAISSVectorStore()
memory = ConversationMemory()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
for d in ["data/uploaded_docs", "temp", "faiss_index"]:
    os.makedirs(d, exist_ok=True)

class ChatRequest(BaseModel):
    query: str
    filter_doc: Optional[str] = None

class TTSRequest(BaseModel):
    text: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document, process it, and add to the vector store.
    """
    file_path = os.path.join("data/uploaded_docs", file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pages = load_document(file_path)
        if not pages:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail="Document contains no extractable text or is too short (min 50 chars).")
            
        chunks = split_into_chunks(pages)
        if not chunks:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail="Document could not be processed into meaningful context.")
            
        vector_store.add_document(chunks, file.filename)
        
        return {
            "doc_name": file.filename,
            "chunks_created": len(chunks),
            "status": "success"
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Text chat endpoint using Gemini.
    """
    try:
        response = get_answer(
            request.query, 
            vector_store, 
            memory, 
            gemini_client=get_gemini_client(),
            filter_doc=request.filter_doc
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice-input")
async def chat_voice_input(audio: UploadFile = File(...)):
    """
    Endpoint for Gemini-powered voice-to-text input + RAG chat.
    """
    try:
        audio_bytes = await audio.read()
        client = get_gemini_client()
        transcription = transcribe_audio(audio_bytes, gemini_client=client)
        
        if not transcription:
            return {"answer": "I couldn't hear you clearly.", "transcription": ""}
            
        response = get_answer(
            transcription, 
            vector_store, 
            memory, 
            gemini_client=client
        )
        response["transcription"] = transcription
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice-output")
async def chat_voice_output(request: TTSRequest):
    """
    Free Text-to-speech endpoint.
    """
    try:
        audio_content = synthesize_speech(request.text)
        return StreamingResponse(io.BytesIO(audio_content), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """
    Get list of documents in vector store.
    """
    return vector_store.get_documents()

@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """
    Delete a document from vector store and disk.
    """
    try:
        vector_store.delete_document(doc_name)
        # Delete file from disk
        file_path = os.path.join("data/uploaded_docs", doc_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "success", "message": f"Deleted {doc_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/clear")
async def clear_chat():
    """
    Clear conversation memory.
    """
    memory.clear()
    return {"status": "cleared"}

@app.get("/chat/history")
async def chat_history():
    """
    Get message history.
    """
    return memory.to_list()

if __name__ == "__main__":
    import uvicorn
    # Use environment port for hosting platforms
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
