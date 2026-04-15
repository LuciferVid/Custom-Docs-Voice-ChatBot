import streamlit as st
import requests
import os
from audio_recorder_streamlit import audio_recorder
import time
import base64

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Voice RAG Chatbot", page_icon="💬", layout="wide")

# Custom CSS for Premium Midnight Stealth Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    /* Midnight Base */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e2e8f0;
    }
    
    /* Premium Glassmorphism Cards */
    .user-bubble {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 18px 22px;
        border-radius: 24px 24px 4px 24px;
        margin-bottom: 24px;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        font-weight: 500;
        animation: fadeInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: #ffffff;
        padding: 18px 22px;
        border-radius: 24px 24px 24px 4px;
        margin-bottom: 24px;
        max-width: 85%;
        float: left;
        clear: both;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        animation: fadeInLeft 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .stMarkdown p {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
    }
    
    /* Sidebar Overhaul */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Buttons and Inputs */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    @keyframes fadeInRight { from { opacity: 0; transform: translateX(30px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes fadeInLeft { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    try:
        resp = requests.get(f"{API_URL}/chat/history")
        if resp.status_code == 200:
            st.session_state.messages = resp.json()
        else:
            st.session_state.messages = []
    except:
        st.session_state.messages = []

if "auto_play" not in st.session_state:
    st.session_state.auto_play = True

def play_audio(text):
    """Fetches and plays TTS audio (Free Version)."""
    try:
        resp = requests.post(f"{API_URL}/chat/voice-output", json={"text": text})
        if resp.status_code == 200:
            audio_base64 = base64.b64encode(resp.content).decode("utf-8")
            st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">', unsafe_allow_html=True)
    except:
        pass

# Sidebar
with st.sidebar:
    st.title("📚 Your Documents")
    
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_URL}/upload", files=files)
                if resp.status_code == 200:
                    st.toast(f"✅ {uploaded_file.name} uploaded successfully!")
                else:
                    st.error(f"❌ Failed to upload {uploaded_file.name}")

    st.divider()
    
    # List documents
    try:
        docs_resp = requests.get(f"{API_URL}/documents")
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    col1.text(f"📄 {doc['doc_name']}")
                    if col2.button("🗑️", key=f"del_{doc['doc_name']}"):
                        requests.delete(f"{API_URL}/documents/{doc['doc_name']}")
                        st.rerun()
            else:
                st.info("No documents uploaded yet.")
    except:
        st.error("Connection failed.")

    st.divider()
    st.subheader("🤖 Intelligence")
    st.info("Powered by Google Gemini 2.0 Flash")
    provider_id = "gemini"
    
    st.divider()
    st.subheader("🔊 Voice Settings")
    st.session_state.auto_play = st.toggle("Auto-play responses", value=st.session_state.auto_play)
    
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        requests.post(f"{API_URL}/chat/clear")
        st.session_state.messages = []
        st.rerun()

# Main Area
st.title("💬 Chat with Your Documents")

# Robust document fetching
try:
    docs_resp = requests.get(f"{API_URL}/documents")
    docs = docs_resp.json() if docs_resp.status_code == 200 else []
except:
    docs = []

if not docs:
    st.warning("👆 Upload documents from the sidebar to get started", icon="⚠️")
else:
    # Document filter
    doc_options = ["All Documents"] + [d['doc_name'] for d in docs]
    selected_doc = st.selectbox("Search in:", doc_options)
    filter_doc = None if selected_doc == "All Documents" else selected_doc

    # Chat Area
    for i, msg in enumerate(st.session_state.messages):
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
            if st.button("🔊 Play", key=f"play_{i}"):
                play_audio(content)

    # Input Area
    st.write("---")
    col1, col2, col3 = st.columns([10, 1, 1], vertical_alignment="bottom")
    with col1:
        user_input = st.text_input("Type your question...", key="text_input", label_visibility="collapsed")
    with col2:
        audio_bytes = audio_recorder(text="", icon_size="2x", neutral_color="#007bff")
    with col3:
        send_button = st.button("Send", use_container_width=True)

    if (send_button and user_input) or audio_bytes:
        payload = {"query": user_input, "filter_doc": filter_doc, "provider": provider_id}
        url = f"{API_URL}/chat"
        
        if audio_bytes:
            files = {"audio": ("query.wav", audio_bytes, "audio/wav")}
            resp = requests.post(f"{API_URL}/chat/voice-input?provider={provider_id}", files=files)
        else:
            resp = requests.post(url, json=payload)
            
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.messages.append({"role": "user", "content": result.get("transcription", user_input)})
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            if st.session_state.auto_play:
                play_audio(result["answer"])
            st.rerun()
