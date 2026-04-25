import streamlit as st
import requests
import os
from audio_recorder_streamlit import audio_recorder
import time
import base64
from dotenv import load_dotenv

import uuid

load_dotenv()

# --- Configuration ---
# Production Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "https://custom-docs-voice-chatbot.onrender.com")

# Session ID Management
params = st.query_params
if "session_id" not in st.session_state:
    if "session" in params:
        st.session_state.session_id = params["session"]
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params["session"] = st.session_state.session_id
elif "session" not in params or params["session"] != st.session_state.session_id:
    st.query_params["session"] = st.session_state.session_id

SESSION_ID = st.session_state.session_id
SESSION_HEADERS = {"X-Session-ID": SESSION_ID}

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
    
    /* HIDE STREAMING PLAYER BUT KEEP IT FUNCTIONAL */
    div[data-testid="stAudio"] { 
        position: fixed; bottom: 0; left: 0; width: 1px; height: 1px; opacity: 0.01; overflow: hidden; pointer-events: none;
    }
    
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] { visibility: visible !important; background: transparent !important; }
    button[data-testid="stSidebarCollapseButton"] { background-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

def play_audio(text):
    """
    Synthesizes speech from text using the backend and plays it in the browser.
    """
    try:
        resp = requests.post(f"{BACKEND_URL}/chat/voice-output", json={"text": text}, headers=SESSION_HEADERS, timeout=30)
        if resp.status_code == 200:
            st.audio(resp.content, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.error(f"Audio Synthesis Failed: {e}")

# Session state initialization
def get_docs():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", headers=SESSION_HEADERS)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []

if "messages" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/chat/history", headers=SESSION_HEADERS, timeout=2)
        st.session_state.messages = resp.json() if resp.status_code == 200 else []
    except:
        st.session_state.messages = []

if "auto_play" not in st.session_state:
    st.session_state.auto_play = True

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Global Document Fetch (High Timeout for Render)
if "docs" not in st.session_state:
    st.session_state.docs = None

try:
    docs_resp = requests.get(f"{BACKEND_URL}/documents?t={time.time()}", headers=SESSION_HEADERS, timeout=15)
    if docs_resp.status_code == 200:
        st.session_state.docs = docs_resp.json()
    else:
        st.session_state.docs = []
except Exception as e:
    st.session_state.docs = None

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "last_doc" not in st.session_state:
    st.session_state.last_doc = None

def sync_intelligence(files_to_sync):
    success_count = 0
    for f in files_to_sync:
        with st.spinner(f"Synchronizing {f.name}..."):
            try:
                # 180s timeout for massive files on Render
                files = {"file": (f.name, f.getvalue())}
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, headers=SESSION_HEADERS, timeout=180)
                if resp.status_code == 200:
                    st.toast(f"✅ Indexed: {f.name}")
                    success_count += 1
                else:
                    err_msg = resp.json().get('detail', 'System Rejected')
                    st.error(f"❌ {f.name} Sync Failed: {err_msg}")
            except Exception as e:
                st.error(f"⚠️ {f.name} Signal Lost: {str(e)}")
            
            # Brief pause to let backend breathe
            time.sleep(0.5)
    return success_count

def process_query(query, is_audio=False, audio_data=None):
    effective_query = query if query and query.strip() else "Please provide a comprehensive summary of the latest intelligence."
    payload = {"query": effective_query, "filter_doc": None}
    
    # Pre-render user message for immediate feedback
    st.session_state.messages.append({"role": "user", "content": effective_query})
    
    # Audio processing remains non-streaming for transcription
    if is_audio and audio_data:
        with st.status("📡 Processing Audio Signal...", expanded=False) as status:
            files = {"audio": ("signal.wav", audio_data, "audio/wav")}
            resp = requests.post(f"{BACKEND_URL}/chat/voice-input", files=files, headers=SESSION_HEADERS, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                # Update user message with actual transcription
                st.session_state.messages[-1]["content"] = result.get("transcription", effective_query)
                answer = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})
                if st.session_state.auto_play:
                    play_audio(answer)
                st.rerun()
            else:
                st.error("Audio signal lost. Please try again.")
                return

    # Text query with STREAMING
    full_answer = ""
    sources = []
    
    with st.status("🧠 Consulting Intelligence Core...", expanded=True) as status:
        try:
            with requests.post(f"{BACKEND_URL}/chat/stream", json=payload, headers=SESSION_HEADERS, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    st.error("Signal Lost: Intelligence core is unreachable.")
                    return
                
                # Create a placeholder for the streaming answer
                answer_placeholder = st.empty()
                
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        
                        if data["type"] == "rephrased":
                            status.update(label=f"🔍 Searching: {data['content']}")
                        
                        elif data["type"] == "sources":
                            sources = data["content"]
                            status.update(label=f"📄 Found {len(sources)} relevant context points.")
                        
                        elif data["type"] == "chunk":
                            full_answer += data["content"]
                            # Preview the answer in a temporary bubble style
                            answer_placeholder.markdown(f"""
                            <div class="assistant-bubble">
                                {full_answer} ▌
                            </div>
                            """, unsafe_allow_html=True)
                            
                        elif data["type"] == "done":
                            full_answer = data["answer"]
                            sources = data["sources"]
                            status.update(label="✅ Analysis Complete", state="complete")
                        
                        elif data["type"] == "error":
                            st.error(f"Intelligence Error: {data['content']}")
                            return

            # Finalize
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            if st.session_state.auto_play:
                play_audio(full_answer)
            st.rerun()
            
        except Exception as e:
            st.error(f"📡 Connection Reset: {str(e)}")

# Sidebar
with st.sidebar:
    st.title("System Control")
    st.caption(f"Backend Status: {'🟢 Online' if st.session_state.docs is not None else '🔴 Offline/Waking Up...'}")
    st.divider()
    
    st.subheader("Document Repository")
    
    # Hide uploader if 3 files are already indexed
    indexed_count = len(st.session_state.docs) if st.session_state.docs else 0
    if indexed_count < 3:
        uploaded_files = st.file_uploader("Index Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    else:
        st.info("📂 **Repository Full**: Please delete a document to upload new intelligence.")
        uploaded_files = None
    
    # Track files for auto-sync and rescue
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("⚠️ **Safety Limit**: Please upload at most 3 documents at a time to prevent API rate-limiting.")
            st.session_state.on_deck = []
        else:
            st.session_state.on_deck = uploaded_files
            current_docs = st.session_state.docs if st.session_state.docs is not None else []
            files_to_sync = [f for f in uploaded_files if not any(d['doc_name'] == f.name for d in current_docs)]
            
            if files_to_sync:
                st.info(f"📡 {len(files_to_sync)} file(s) ready for initialization.")
                if st.button("🚀 Sync to Intelligence", use_container_width=True):
                    success_count = sync_intelligence(files_to_sync)
                    if success_count > 0:
                        time.sleep(1)
                        st.rerun()
    else:
        st.session_state.on_deck = []

    st.divider()
    
    # Active Intelligence Context
    if st.session_state.docs:
        st.caption("Active Intelligence Context:")
        for doc in st.session_state.docs:
            col1, col2 = st.columns([5, 1])
            chunks = doc.get('chunk_count', 0)
            col1.markdown(f"🔹 **{doc['doc_name']}**")
            col1.caption(f"📡 {chunks} Intelligence Chunks Digested")
            if col2.button("🗑️", key=f"del_{doc['doc_name']}"):
                requests.delete(f"{BACKEND_URL}/documents/{doc['doc_name']}", headers=SESSION_HEADERS, timeout=15)
                st.rerun()
    else:
        st.caption("No intelligence context currently loaded.")

    st.divider()
    st.subheader("System Safety")
    st.session_state.auto_play = st.toggle("Voice Synthesis Output", value=st.session_state.auto_play)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset Chat", use_container_width=True):
            try: requests.post(f"{BACKEND_URL}/chat/clear", headers=SESSION_HEADERS, timeout=10)
            except: pass
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Hard Reset", use_container_width=True, help="Purge all indexed documents and clear memory"):
            try:
                # Delete all docs found
                for doc in st.session_state.docs:
                    requests.delete(f"{BACKEND_URL}/documents/{doc['doc_name']}", headers=SESSION_HEADERS, timeout=10)
                requests.post(f"{BACKEND_URL}/chat/clear", headers=SESSION_HEADERS, timeout=10)
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
            s_resp = requests.get(f"{BACKEND_URL}/documents/{current_latest_doc}/suggestions?t={time.time()}", headers=SESSION_HEADERS)
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

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

def on_enter():
    if st.session_state.text_input:
        st.session_state.pending_query = st.session_state.text_input
        st.session_state.text_input = ""

c1, c2, c3 = st.columns([12, 1, 2], vertical_alignment="bottom")
with c1:
    st.text_input("Summarize, ask a question, or find insights...", key="text_input", label_visibility="collapsed", on_change=on_enter, placeholder="Type here or leave empty for a full summary...")
with c2:
    audio_stream = audio_recorder(text="", icon_size="2x", neutral_color="#94a3b8")
with c3:
    send_trigger = st.button("Analyze", use_container_width=True, type="primary")

query_to_process = None
if send_trigger:
    # Handle button click - ALLOW empty for summary
    query_to_process = st.session_state.text_input
    st.session_state.text_input = ""
    # If both are empty, it will trigger the default summary in process_query
elif st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

if query_to_process is not None:
    process_query(query_to_process)
elif audio_stream and audio_stream != st.session_state.last_audio_bytes:
    st.session_state.last_audio_bytes = audio_stream
    process_query("", is_audio=True, audio_data=audio_stream)
