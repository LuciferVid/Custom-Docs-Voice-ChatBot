import streamlit as st
import requests
import os
from audio_recorder_streamlit import audio_recorder
import time
import base64

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Voice RAG Chatbot", page_icon="💬", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #f8f9fa, #e9ecef);
    }
    
    /* Premium Chat Bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        color: white;
        padding: 14px 18px;
        border-radius: 20px 20px 4px 20px;
        margin-bottom: 20px;
        max-width: 85%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        animation: fadeInRight 0.3s ease-out;
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        color: #1f2937;
        padding: 14px 18px;
        border-radius: 20px 20px 20px 4px;
        margin-bottom: 20px;
        max-width: 85%;
        float: left;
        clear: both;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        animation: fadeInLeft 0.3s ease-out;
    }
    
    .source-tag {
        font-size: 0.85em;
        color: #4b5563;
        font-weight: 600;
        margin-top: 8px;
    }
    
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
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

def play_audio(text, voice):
    """Fetches and plays TTS audio."""
    try:
        resp = requests.post(f"{API_URL}/chat/voice-output", json={"text": text, "voice": voice})
        if resp.status_code == 200:
            audio_base64 = base64.b64encode(resp.content).decode("utf-8")
            audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
            st.markdown(audio_tag, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error playing audio: {e}")

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
                    col1.text(f"📄 {doc['doc_name']} ({doc['chunk_count']} chunks)")
                    if col2.button("🗑️", key=f"del_{doc['doc_name']}"):
                        requests.delete(f"{API_URL}/documents/{doc['doc_name']}")
                        st.rerun()
            else:
                st.info("No documents uploaded yet.")
    except:
        st.error("Cannot connect to backend server.")

    st.divider()
    st.subheader("🔊 Voice Settings")
    voice_option = st.selectbox("Select Voice", ["nova", "alloy", "echo", "onyx", "shimmer"], format_func=lambda x: x.capitalize())
    st.session_state.auto_play = st.toggle("Auto-play responses", value=st.session_state.auto_play)
    
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        requests.post(f"{API_URL}/chat/clear")
        st.session_state.messages = []
        st.rerun()

# Main Area
st.title("💬 Chat with Your Documents")

if not any(requests.get(f"{API_URL}/documents").json()):
    st.warning("👆 Upload documents from the sidebar to get started", icon="⚠️")
else:
    # Document filter
    docs = requests.get(f"{API_URL}/documents").json()
    doc_options = ["All Documents"] + [d['doc_name'] for d in docs]
    selected_doc = st.selectbox("Search in:", doc_options)
    filter_doc = None if selected_doc == "All Documents" else selected_doc

    # Chat Area
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            content = msg["content"]
            metadata = msg.get("metadata", {})
            
            if role == "user":
                st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
                
                # Show sources if available
                sources = metadata.get("sources", [])
                if sources:
                    with st.expander("📄 Sources", expanded=False):
                        for src in sources:
                            st.caption(f"- {src}")
                
                # Manual Play Button
                if st.button("🔊 Play Response", key=f"play_{i}"):
                    play_audio(content, voice_option)

    # Sticky Input Area at the bottom
    st.write("---")
    col1, col2, col3 = st.columns([10, 1, 1], vertical_alignment="bottom")
    
    with col1:
        user_input = st.text_input("Type your question...", key="text_input", label_visibility="collapsed")
    
    with col2:
        audio_bytes = audio_recorder(text="", icon_size="2x", neutral_color="#007bff")
    
    with col3:
        send_button = st.button("Send", use_container_width=True)

    # Logic for text input
    if send_button and user_input:
        payload = {"query": user_input, "filter_doc": filter_doc}
        with st.spinner("Thinking..."):
            resp = requests.post(f"{API_URL}/chat", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                
                if st.session_state.auto_play:
                    play_audio(result["answer"], voice_option)
                st.rerun()

    # Logic for audio input
    if audio_bytes:
        with st.spinner("Transcribing..."):
            files = {"audio": ("query.wav", audio_bytes, "audio/wav")}
            resp = requests.post(f"{API_URL}/chat/voice-input", files=files)
            if resp.status_code == 200:
                result = resp.json()
                if result.get("transcription"):
                    st.session_state.messages.append({"role": "user", "content": result["transcription"]})
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    
                    if st.session_state.auto_play:
                        play_audio(result["answer"], voice_option)
                    st.rerun()
                else:
                    st.warning("Could not transcribe audio.")
