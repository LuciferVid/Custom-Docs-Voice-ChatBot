import io
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)

def synthesize_speech(text: str) -> bytes:
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
