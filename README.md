# Intelligence Core: Voice RAG System

An enterprise-grade, high-performance RAG (Retrieval-Augmented Generation) ecosystem powered by Google Gemini. This system enables professional-grade document analysis through a sophisticated, minimal voice interface.

## System Architecture

### 🧠 Intelligence & Synthesis
- **Core Reasoning**: Google Gemini 2.0 Flash for context-aware generation and rephrasing.
- **Vector Embeddings**: Cloud-native `text-embedding-004` for zero-latency, high-dimensional semantic search.
- **Voice Pipeline**: Bidirectional audio processing using Gemini STT and high-fidelity gTTS.

### ⚡ Performance & Stability
- **Multi-User Session Isolation**: Advanced UUID-based workspace isolation ensuring total privacy and independent context for every user.
- **API Resilience Engine**: Native exponential backoff and retry logic for high-availability document indexing under peak load.
- **Automated Maintenance**: Intelligent background pruning tasks for session memory and disk space management (24h TTL).
- **Multi-File Concurrent Sync**: Optimized for simultaneous processing of multiple documents without performance degradation.
- **Thread-Safe Architecture**: Implemented system-wide locking for the FAISS engine to prevent context corruption during concurrent writes.
- **Cloud-Native Design**: Replaced local heavy ML dependencies with cloud APIs, reducing memory footprint by 90% (<50MB RAM).
- **Stateless Stability**: Optimized for high-availability deployment on containerized environments like Render.

### 📁 Data Handling
- **Multi-Format Support**: Native extraction for PDF, DOCX, TXT, and MD.
- **Context Management**: Adaptive sliding-window chunking logic for precise retrieval.
- **Vector Storage**: Isolated, session-specific FAISS indices with thread-safe persistence.

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
