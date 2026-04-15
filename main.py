import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
from openai import OpenAI
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

# Initialize global instances
app = FastAPI(title="Voice RAG Chatbot API")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize Gemini client with new SDK
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
    provider: str = "openai"  # "openai" or "gemini"

class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"

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
        chunks = split_into_chunks(pages)
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
    Text chat endpoint.
    """
    try:
        response = get_answer(
            request.query, 
            vector_store, 
            memory, 
            openai_client, 
            gemini_client=gemini_client,
            provider=request.provider,
            filter_doc=request.filter_doc
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice-input")
async def chat_voice_input(audio: UploadFile = File(...), provider: str = "openai"):
    """
    Endpoint for voice-to-text input + RAG chat.
    """
    try:
        audio_bytes = await audio.read()
        transcription = transcribe_audio(openai_client, audio_bytes, gemini_client=gemini_client)
        
        if not transcription:
            return {"answer": "I couldn't hear you clearly.", "transcription": ""}
            
        response = get_answer(
            transcription, 
            vector_store, 
            memory, 
            openai_client, 
            gemini_client=gemini_client,
            provider=provider
        )
        response["transcription"] = transcription
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/voice-output")
async def chat_voice_output(request: TTSRequest):
    """
    Text-to-speech endpoint.
    """
    try:
        audio_content = synthesize_speech(openai_client, request.text, request.voice)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
