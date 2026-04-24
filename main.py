# Voice Chatbot with RAG - Version 1.2.0 (Groq Priority)
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from ingestion.document_loader import load_document
from ingestion.text_splitter import split_into_chunks
from vector_store.faiss_store import FAISSVectorStore
from prompts.templates import SUGGEST_PROMPT
import json
from rag.memory import ConversationMemory
from rag.chain import get_answer
from voice.speech_to_text import transcribe_audio
from voice.text_to_speech import synthesize_speech

app = FastAPI(title="Voice RAG Chatbot API")

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

from fastapi import Header

# --- Session Management ---
_sessions = {}

def get_state(session_id: str):
    if not session_id:
        session_id = "default"
    if session_id not in _sessions:
        session_dir = f"faiss_index/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        _sessions[session_id] = {
            "vector_store": FAISSVectorStore(index_dir=session_dir),
            "memory": ConversationMemory()
        }
    return _sessions[session_id]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure base directories exist
for d in ["data/uploaded_docs", "temp", "faiss_index"]:
    os.makedirs(d, exist_ok=True)

class ChatRequest(BaseModel):
    query: str
    filter_doc: Optional[str] = None

class TTSRequest(BaseModel):
    text: str

@app.post("/upload")
def upload_document(file: UploadFile = File(...), x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    v_store = state["vector_store"]
    
    file_path = os.path.join("data/uploaded_docs", f"{x_session_id}_{file.filename}")
    print(f"Received upload request for: {file.filename} (Session: {x_session_id})")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Loading document: {file.filename}")
        pages = load_document(file_path)
        if not pages:
            if os.path.exists(file_path): os.remove(file_path)
            raise HTTPException(status_code=400, detail="Document contains no extractable text.")
        
        chunks = split_into_chunks(pages)
        v_store.add_document(chunks, file.filename)
        
        return {"doc_name": file.filename, "status": "success"}
    except Exception as e:
        print(f"Upload failed: {e}")
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest, x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    try:
        response = get_answer(
            request.query, 
            state["vector_store"], 
            state["memory"], 
            groq_client=get_groq_client(),
            filter_doc=request.filter_doc
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice-input")
async def chat_voice_input(audio: UploadFile = File(...), x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    try:
        audio_bytes = await audio.read()
        client = get_groq_client()
        transcription = transcribe_audio(audio_bytes, groq_client=client)
        if not transcription:
            return {"answer": "I couldn't hear you clearly.", "transcription": ""}
        response = get_answer(transcription, state["vector_store"], state["memory"], groq_client=client)
        response["transcription"] = transcription
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/voice-output")
def chat_voice_output(request: TTSRequest):
    # Voice output is stateless, no session needed
    try:
        audio_content = synthesize_speech(request.text)
        return StreamingResponse(io.BytesIO(audio_content), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    return state["vector_store"].get_documents()

@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str, x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    state["vector_store"].delete_document(doc_name)
    file_path = os.path.join("data/uploaded_docs", f"{x_session_id}_{doc_name}")
    if os.path.exists(file_path): os.remove(file_path)
    return {"status": "success"}

@app.post("/chat/clear")
async def clear_chat(x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    state["memory"].clear()
    return {"status": "cleared"}

@app.get("/chat/history")
async def chat_history(x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    return state["memory"].to_list()

@app.get("/documents/{doc_name}/suggestions")
def get_suggestions(doc_name: str, x_session_id: str = Header(None)):
    state = get_state(x_session_id)
    v_store = state["vector_store"]
    try:
        relevant_chunks = [c["text"] for c in v_store.chunks if c.get("source_file") == doc_name][:3]
        if not relevant_chunks: return ["Summarize this document"]
        context = "\n---\n".join(relevant_chunks)
        prompt = SUGGEST_PROMPT.format(context=context)
        client = get_groq_client()
        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
        text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(text)[:3]
    except:
        return ["Summarize context", "Key takeaways", "Main points"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
