# 🎙️ Custom Docs Voice ChatBot (Gemini Edition)

A premium, high-performance RAG (Retrieval-Augmented Generation) chatbot that allows you to talk to your documents using voice and high-fidelity intelligence.

## ✨ Features
- **🧠 100% Gemini Powered**: Uses Google's latest model for all reasoning and transcription.
- **🔊 Voice-First Interaction**: High-quality free speech synthesis and native audio understanding.
- **📁 Document Intelligence**: Support for PDF, DOCX, TXT, and MD files.
- **🌑 Midnight Stealth UI**: A sophisticated, glassmorphism dark-mode interface.
- **⚡ Supercharged RAG**: Native, lightweight text splitting and FAISS vector storage.

## 🛠️ Tech Stack
- **AI Engine**: Google Gemini 2.0 Flash
- **Voice Synthesis**: Google gTTS
- **Vector DB**: FAISS (Meta)
- **Framework**: FastAPI (Backend) & Streamlit (Frontend)
- **Embeddings**: Sentence-Transformers

## 🚀 Quick Start

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
**Start the Backend:**
```bash
python main.py
```

**Start the Frontend:**
```bash
streamlit run frontend/app.py
```

## 🌐 Deployment
This project is optimized for deployment on **Render** (Backend) and **Streamlit Cloud** or **Hugging Face** (Frontend). Ensure the `GEMINI_API_KEY` is set in your hosting environment.

---
Created by [LuciferVid](https://github.com/LuciferVid)
