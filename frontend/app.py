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
def get_docs():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents")
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []

if "messages" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/chat/history", timeout=2)
        st.session_state.messages = resp.json() if resp.status_code == 200 else []
    except:
        st.session_state.messages = []

if "auto_play" not in st.session_state:
    st.session_state.auto_play = True

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Global Document Fetch (High Timeout for Render)
try:
    docs_resp = requests.get(f"{BACKEND_URL}/documents?t={time.time()}", timeout=15)
    st.session_state.docs = docs_resp.json() if docs_resp.status_code == 200 else []
except Exception as e:
    st.session_state.docs = []
    st.sidebar.error(f"Signal Timeout: Backend is waking up... {str(e)}")

if "currently_syncing" not in st.session_state:
    st.session_state.currently_syncing = set()

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "last_doc" not in st.session_state:
    st.session_state.last_doc = None

def play_audio(text):
    try:
        resp = requests.post(f"{BACKEND_URL}/chat/voice-output", json={"text": text}, timeout=15)
        if resp.status_code == 200:
            audio_base64 = base64.b64encode(resp.content).decode("utf-8")
            st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">', unsafe_allow_html=True)
    except:
        pass

def process_query(query, is_audio=False, audio_data=None):
    effective_query = query if query and query.strip() else "Please provide a comprehensive summary of the latest intelligence."
    payload = {"query": effective_query, "filter_doc": None}
    
    if is_audio and audio_data:
        with st.spinner("Processing Signal..."):
            files = {"audio": ("signal.wav", audio_data, "audio/wav")}
            resp = requests.post(f"{BACKEND_URL}/chat/voice-input", files=files, timeout=15)
    else:
        with st.spinner("Analyzing Intelligence..." if not (query and query.strip()) else "Finding Answer..."):
            resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=15)
        
    if resp.status_code == 200:
        result = resp.json()
        st.session_state.messages.append({"role": "user", "content": result.get("transcription", effective_query)})
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        if st.session_state.auto_play:
            play_audio(result["answer"])
        st.rerun()

# Sidebar
with st.sidebar:
    st.title("System Control")
    st.caption(f"Backend Status: {'🟢 Online' if st.session_state.docs is not None else '🔴 Connecting...'}")
    st.divider()
    
    st.subheader("Document Repository")
    uploaded_files = st.file_uploader("Index Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        files_to_sync = [f for f in uploaded_files if not any(d['doc_name'] == f.name for d in st.session_state.docs)]
        
        if files_to_sync:
            st.warning(f"📡 {len(files_to_sync)} file(s) awaiting intelligence sync.")
            if st.button("🚀 Sync to Intelligence", use_container_width=True):
                for f in files_to_sync:
                    with st.spinner(f"Synchronizing {f.name}..."):
                        try:
                            # Explicitly check for empty content
                            content = f.getvalue()
                            if not content:
                                st.error(f"❌ {f.name} is empty.")
                                continue
                                
                            files = {"file": (f.name, content)}
                            resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                            if resp.status_code == 200:
                                st.toast(f"✅ Indexed: {f.name}")
                            else:
                                err_msg = resp.json().get('detail', 'System Rejected')
                                st.error(f"❌ Failed: {err_msg}")
                        except Exception as e:
                            st.error(f"⚠️ Network Signal Lost: {str(e)}")
                st.rerun()

    st.divider()
    
    # Active Intelligence Context
    if st.session_state.docs:
        st.caption("Active Intelligence Context:")
        for doc in st.session_state.docs:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"🔹 **{doc['doc_name']}**")
            if col2.button("🗑️", key=f"del_{doc['doc_name']}"):
                requests.delete(f"{BACKEND_URL}/documents/{doc['doc_name']}", timeout=15)
                st.rerun()
    else:
        st.caption("No intelligence context currently loaded.")

    st.divider()
    st.subheader("System Safety")
    st.session_state.auto_play = st.toggle("Voice Synthesis Output", value=st.session_state.auto_play)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset Chat", use_container_width=True):
            try: requests.post(f"{BACKEND_URL}/chat/clear", timeout=10)
            except: pass
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Hard Reset", use_container_width=True, help="Purge all indexed documents and clear memory"):
            try:
                # Delete all docs found
                for doc in st.session_state.docs:
                    requests.delete(f"{BACKEND_URL}/documents/{doc['doc_name']}", timeout=10)
                requests.post(f"{BACKEND_URL}/chat/clear", timeout=10)
            except: pass
            st.session_state.messages = []
            st.session_state.docs = []
            st.rerun()

# Main Interface
st.header("Intelligence Interface")

# Universal Intelligence Logic (No Manual Scope Required)
if "last_doc" not in st.session_state:
    st.session_state.last_doc = None

# Automatically fetch suggestions for the latest document added
current_latest_doc = st.session_state.docs[0]['doc_name'] if st.session_state.docs else None

if current_latest_doc != st.session_state.last_doc:
    st.session_state.last_doc = current_latest_doc
    if current_latest_doc:
        try:
            s_resp = requests.get(f"{BACKEND_URL}/documents/{current_latest_doc}/suggestions?t={time.time()}")
            st.session_state.suggestions = s_resp.json() if s_resp.status_code == 200 else []
        except:
            st.session_state.suggestions = []
    else:
        st.session_state.suggestions = []

# Display Suggestions
if st.session_state.suggestions:
    st.caption("Suggested Investigations:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion, key=f"sug_{i}", use_container_width=True):
            process_query(suggestion)

# Chat Thread
if not st.session_state.messages:
    st.markdown("### 📥 Document Signal Received")
    st.info("I am ready to analyze your context. You can use the signal bar below to:")
    c1, c2 = st.columns(2)
    with c1:
        st.write("📝 **Ask specific questions**")
        st.caption("e.g., 'What is the contract duration?'")
    with c2:
        st.write("📊 **Request a summary**")
        st.caption("Just click 'Analyze' with an empty bar.")
else:
    for i, msg in enumerate(st.session_state.messages):
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
            if st.button("Listen", key=f"play_{i}"):
                play_audio(content)

# Input Control (Always Visible)
st.write("---")
c1, c2, c3 = st.columns([10, 1, 1], vertical_alignment="bottom")
with c1:
    user_input = st.text_input("Summarize, ask a question, or find insights...", key="text_input", label_visibility="collapsed")
with c2:
    audio_stream = audio_recorder(text="", icon_size="2x", neutral_color="#2563eb")
with c3:
    send_trigger = st.button("Analyze", use_container_width=True)

if (send_trigger) or audio_stream:
    process_query(user_input, is_audio=True if audio_stream else False, audio_data=audio_stream)
