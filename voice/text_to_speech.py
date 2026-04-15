import io
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)

def synthesize_speech(openai_client, text: str, voice: str = "nova") -> bytes:
    """
    Synthesizes speech from text using Google TTS (Free).
    """
    # Trim to 4000 characters
    trimmed_text = text[:4000]
    
    try:
        # Use gTTS for free, reliable high-quality voice
        tts = gTTS(text=trimmed_text, lang='en', slow=False)
        
        # Save to bytes stream
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp.getvalue()
        
    except Exception as e:
        logger.error(f"Error synthesizing speech with gTTS: {e}")
        return b""

def save_audio(audio_bytes: bytes, filename: str = None) -> str:
    """
    Saves audio bytes to a temp file and returns the path.
    """
    if not filename:
        filename = f"speech_{uuid.uuid4()}.mp3"
        
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    file_path = os.path.join("temp", filename)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path
