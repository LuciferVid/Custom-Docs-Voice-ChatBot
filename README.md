# Intelligence Core: Voice RAG System

An enterprise-grade, high-performance RAG (Retrieval-Augmented Generation) ecosystem powered by Google Gemini. This system enables professional-grade document analysis through a sophisticated, minimal voice interface.

## System Architecture

### 🧠 Intelligence & Synthesis
- **Core Reasoning**: Google Gemini 2.0 Flash for context-aware generation and rephrasing.
- **Vector Embeddings**: Cloud-native `text-embedding-004` for zero-latency, high-dimensional semantic search.
- **Voice Pipeline**: Bidirectional audio processing using Gemini STT and high-fidelity gTTS.

### ⚡ Performance & Stability
- **Multi-File Concurrent Sync**: Optimized for simultaneous processing of 4+ documents without performance degradation.
- **Thread-Safe Architecture**: Implemented system-wide locking for the FAISS engine to prevent context corruption during concurrent writes.
- **API Batch Processing**: Intelligent batching logic for Google Gemini API to handle massive document ingestion within cloud limits.
- **Cloud-Native Design**: Replaced local heavy ML dependencies with cloud APIs, reducing memory footprint by 90% (<50MB RAM).
- **Stateless Stability**: Optimized for high-availability deployment on containerized environments like Render.
- **Lazy Initialization**: Implemented deferred client loading to ensure near-instant platform startup.

### 📁 Data Handling
- **Multi-Format Support**: Native extraction for PDF, DOCX, TXT, and MD.
- **Context Management**: Adaptive sliding-window chunking logic for precise retrieval.
- **Vector Storage**: Proximity-based search using the Meta FAISS engine with thread-safe persistence.

## Quick Connection

### 1. Requirements
Create a `.env` configuration file:
```env
GEMINI_API_KEY=your_production_key
```

### 2. Infrastructure Setup
```bash
pip install -r requirements.txt
```

### 3. Execution
**Engine (Backend):**
```bash
python main.py
```

**Interface (Frontend):**
```bash
streamlit run frontend/app.py
```

## Production Deployment
The system is architected for zero-configuration deployment:
- **Backend**: Render (Python/Uvicorn)
- **Frontend**: Streamlit Community Cloud

---
**Technical Documentation by [LuciferVid](https://github.com/LuciferVid)**
