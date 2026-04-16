import streamlit as st
import requests
import os
from audio_recorder_streamlit import audio_recorder
import time
import base64

# --- Configuration ---
# Production Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "https://custom-docs-voice-chatbot.onrender.com")

st.set_page_config(page_title="Intelligence Core | RAG", page_icon="🔗", layout="wide")

# Premium High-End CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0b0e14; color: #f1f5f9; }
    
    .user-bubble {
        background: #2563eb; color: white; padding: 14px 18px;
        border-radius: 16px 16px 4px 16px; margin-bottom: 20px;
        max-width: 75%; float: right; clear: both; font-size: 0.95rem;
    }
    
    .assistant-bubble {
        background: #1e293b; color: #f8fafc; padding: 14px 18px;
        border-radius: 16px 16px 16px 4px; margin-bottom: 20px;
        max-width: 80%; float: left; clear: both; border: 1px solid #334155; font-size: 0.95rem;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important; border-right: 1px solid #1e293b;
    }
    
    .stButton>button {
        background-color: #2563eb; color: white; border: none;
        border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/chat/history", timeout=2)
        st.session_state.messages = resp.json() if resp.status_code == 200 else []
    except:
        st.session_state.messages = []

if "auto_play" not in st.session_state:
    st.session_state.auto_play = True

def play_audio(text):
    try:
        resp = requests.post(f"{BACKEND_URL}/chat/voice-output", json={"text": text}, timeout=10)
        if resp.status_code == 200:
            audio_base64 = base64.b64encode(resp.content).decode("utf-8")
            st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">', unsafe_allow_html=True)
    except:
        pass

# Sidebar
with st.sidebar:
    st.title("System Control")
    st.divider()
    st.subheader("Document Repository")
    
    uploaded_files = st.file_uploader("Index Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Prevent re-uploading if already in the list
            if any(d['doc_name'] == uploaded_file.name for d in (docs if 'docs' in locals() else [])):
                continue
                
            with st.spinner(f"Synchronizing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    # Higher timeout for model loading on Render
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                    if resp.status_code == 200:
                        st.toast(f"Success: {uploaded_file.name}")
                        st.rerun()
                    else:
                        error_detail = "Unknown Error"
                        try:
                            error_detail = resp.json().get("detail", "Unknown backend error")
                        except: pass
                        st.error(f"Indexing Failed: {error_detail}")
                        st.caption(f"File: {uploaded_file.name}")
                except Exception as e:
                    st.error("Engine Connection Offline")
                    st.caption("The backend is likely still deploying or waking up. Please wait 1 minute.")

    st.divider()
    
    # Automatic Document List
    try:
        docs_resp = requests.get(f"{BACKEND_URL}/documents", timeout=2)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([5, 1])
                    col1.caption(f" {doc['doc_name']}")
                    if col2.button("×", key=f"del_{doc['doc_name']}"):
                        requests.delete(f"{BACKEND_URL}/documents/{doc['doc_name']}")
                        st.rerun()
            else:
                st.caption("No documents currently indexed.")
    except:
        st.error("Engine Connection Offline")

    st.divider()
    st.subheader("Configuration")
    st.session_state.auto_play = st.toggle("Voice Synthesis Output", value=st.session_state.auto_play)
    
    if st.button("Reset Session", use_container_width=True):
        try:
            requests.post(f"{BACKEND_URL}/chat/clear")
        except: pass
        st.session_state.messages = []
        st.rerun()

# Main Interface
st.header("Intelligence Interface")

# Context Selection
try:
    docs_resp = requests.get(f"{BACKEND_URL}/documents", timeout=2)
    docs = docs_resp.json() if docs_resp.status_code == 200 else []
except:
    docs = []

if not docs:
    st.info("Awaiting document upload for context initialization.")
else:
    doc_options = ["Universal Context"] + [d['doc_name'] for d in docs]
    selected_doc = st.selectbox("Analysis Scope", doc_options)
    filter_doc = None if selected_doc == "Universal Context" else selected_doc

    # Chat Thread
    for i, msg in enumerate(st.session_state.messages):
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
            if st.button("Listen", key=f"play_{i}"):
                play_audio(content)

    # Input Control
    st.write("---")
    c1, c2, c3 = st.columns([10, 1, 1], vertical_alignment="bottom")
    with c1:
        user_input = st.text_input("Consult internal knowledge...", key="text_input", label_visibility="collapsed")
    with c2:
        audio_stream = audio_recorder(text="", icon_size="2x", neutral_color="#2563eb")
    with c3:
        send_trigger = st.button("Analyze", use_container_width=True)

    if (send_trigger and user_input) or audio_stream:
        payload = {"query": user_input, "filter_doc": filter_doc}
        
        if audio_stream:
            with st.spinner("Processing Signal..."):
                files = {"audio": ("signal.wav", audio_stream, "audio/wav")}
                resp = requests.post(f"{BACKEND_URL}/chat/voice-input", files=files)
        else:
            with st.spinner("Analyzing..."):
                resp = requests.post(f"{BACKEND_URL}/chat", json=payload)
            
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.messages.append({"role": "user", "content": result.get("transcription", user_input)})
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            if st.session_state.auto_play:
                play_audio(result["answer"])
            st.rerun()
