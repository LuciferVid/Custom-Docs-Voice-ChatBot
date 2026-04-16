import streamlit as st
import requests
import os
from audio_recorder_streamlit import audio_recorder
import time
import base64

# Configuration
DEFAULT_API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if "API_URL" not in st.session_state:
    st.session_state.API_URL = DEFAULT_API_URL

API_URL = st.session_state.API_URL

st.set_page_config(page_title="Intelligence Core | RAG", page_icon="🔗", layout="wide")

# Premium High-End CSS (Lucide & Custom Minimalist)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Elegant Dark Canvas */
    .stApp {
        background-color: #0b0e14;
        color: #f1f5f9;
    }
    
    /* Professional Chat Bubbles */
    .user-bubble {
        background: #2563eb;
        color: white;
        padding: 14px 18px;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 20px;
        max-width: 75%;
        float: right;
        clear: both;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);
    }
    
    .assistant-bubble {
        background: #1e293b;
        color: #f8fafc;
        padding: 14px 18px;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 20px;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Sidebar Sophistication */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Minimalist Inputs */
    .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }

    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    try:
        resp = requests.get(f"{API_URL}/chat/history", timeout=2)
        st.session_state.messages = resp.json() if resp.status_code == 200 else []
    except:
        st.session_state.messages = []

if "auto_play" not in st.session_state:
    st.session_state.auto_play = True

def play_audio(text):
    try:
        resp = requests.post(f"{API_URL}/chat/voice-output", json={"text": text}, timeout=10)
        if resp.status_code == 200:
            audio_base64 = base64.b64encode(resp.content).decode("utf-8")
            st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">', unsafe_allow_html=True)
    except:
        pass

# Sidebar
with st.sidebar:
    st.title("System Console")
    
    # Elegant Connection Manager
    with st.expander("Network Configuration", expanded=(st.session_state.API_URL == "http://localhost:8000")):
        new_url = st.text_input("Backend Endpoint", value=st.session_state.API_URL)
        if new_url != st.session_state.API_URL:
            st.session_state.API_URL = new_url
            st.rerun()
        
        try:
            requests.get(f"{st.session_state.API_URL}/documents", timeout=1)
            st.success("Synchronized")
        except:
            st.error("Offline")
            st.caption("Enter the production Backend Endpoint above.")

    st.divider()
    st.subheader("Knowledge Repository")
    
    uploaded_files = st.file_uploader("Index Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_URL}/upload", files=files)
                if resp.status_code == 200:
                    st.toast(f"Indexed: {uploaded_file.name}")
                else:
                    st.error(f"Error: {uploaded_file.name}")

    st.divider()
    
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=2)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([5, 1])
                    col1.caption(f" {doc['doc_name']}")
                    if col2.button("×", key=f"del_{doc['doc_name']}"):
                        requests.delete(f"{API_URL}/documents/{doc['doc_name']}")
                        st.rerun()
            else:
                st.caption("No active documents indexed.")
    except:
        pass

    st.divider()
    st.subheader("Voice Parameters")
    st.session_state.auto_play = st.toggle("Voice Synthesis Output", value=st.session_state.auto_play)
    
    st.divider()
    if st.button("Initialize Fresh Session", use_container_width=True):
        requests.post(f"{API_URL}/chat/clear")
        st.session_state.messages = []
        st.rerun()

# Main Interface
st.header("Intelligence Interface")

# Document availability check
try:
    docs_resp = requests.get(f"{API_URL}/documents", timeout=2)
    docs = docs_resp.json() if docs_resp.status_code == 200 else []
except:
    docs = []

if not docs:
    st.warning("Please index documents in the System Console to begin analysis.")
else:
    doc_options = ["Universal Context"] + [d['doc_name'] for d in docs]
    selected_doc = st.selectbox("Focus Scope", doc_options)
    filter_doc = None if selected_doc == "Universal Context" else selected_doc

    # Communication Thread
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
        audio_bytes = audio_recorder(text="", icon_size="2x", neutral_color="#2563eb")
    with c3:
        if st.button("Run", use_container_width=True) or (user_input and not audio_bytes):
            pass # Logic handled below

    if (user_input and not audio_bytes and st.session_state.get('last_input') != user_input) or audio_bytes:
        st.session_state.last_input = user_input
        payload = {"query": user_input, "filter_doc": filter_doc}
        
        if audio_bytes:
            with st.spinner("Processing Signal..."):
                files = {"audio": ("signal.wav", audio_bytes, "audio/wav")}
                resp = requests.post(f"{API_URL}/chat/voice-input", files=files)
        else:
            with st.spinner("Analyzing..."):
                resp = requests.post(f"{API_URL}/chat", json=payload)
            
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.messages.append({"role": "user", "content": result.get("transcription", user_input)})
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            if st.session_state.auto_play:
                play_audio(result["answer"])
            st.rerun()
