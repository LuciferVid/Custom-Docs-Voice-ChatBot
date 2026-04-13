# Voice RAG Chatbot - Chat with Your Documents 🎙️📚

A full-stack RAG (Retrieval-Augmented Generation) chatbot that allows users to chat with their documents using text or voice input.

## 🌟 Features

- **Multi-format Support**: Upload PDF, DOCX, TXT, and MD files.
- **Smart RAG**: Standalone query rephrasing with conversation history and context-aware retrieval.
- **Voice Interface**: 
    - **Speech-to-Text**: Record your questions using OpenAI Whisper.
    - **Text-to-Speech**: Listen to responses with premium OpenAI TTS voices.
- **Vector Database**: High-performance local search using FAISS and Sentence Transformers.
- **Persistent Storage**: Documents and FAISS index persist between restarts.
- **Clean UI**: Modern Streamlit interface with chat bubbles and document management.

## 🏗️ Architecture Flow

```text
User Input (Voice/Text) 
      👇
[If Voice] -> Whisper API (STT) -> Transcription
      👇
Retriever (Standalone Query Generation) -> FAISS Search -> Context Chunks
      👇
LLM (GPT-4o) -> Context + Question -> Answer
      👇
[If Auto-play] -> OpenAI TTS (Voice) -> Audio Output
      👇
User Interface (Streamlit)
```

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.9+
- OpenAI API Key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and add your OpenAI API Key:
```
OPENAI_API_KEY=your_sk_...
```

### 4. Run the Backend
```bash
python main.py
```
Backend runs on `http://localhost:8000`.

### 5. Run the Frontend
```bash
streamlit run frontend/app.py
```
Frontend runs on `http://localhost:8501`.

## 📂 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload and process a document |
| `/chat` | POST | Text-based chat with context |
| `/chat/voice-input` | POST | Audio-based chat |
| `/chat/voice-output` | POST | Convert text to speech |
| `/documents` | GET | List all indexed documents |
| `/documents/{name}`| DELETE | Remove a document and re-index |
| `/chat/history` | GET | Retrieve conversation logs |
| `/chat/clear` | POST | Reset memory |

## 🛠️ Tech Stack
- **Backend**: FastAPI, OpenAI SDK, FAISS, LangChain.
- **Frontend**: Streamlit, Audio Recorder.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Models**: `gpt-4o` (Chat), `whisper-1` (STT), `tts-1` (TTS).

---
Built with ❤️ by Antigravity.
