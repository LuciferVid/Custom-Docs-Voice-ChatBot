import os
import uuid
import logging

logger = logging.getLogger(__name__)

def synthesize_speech(openai_client, text: str, voice: str = "nova") -> bytes:
    """
    Synthesizes speech from text using OpenAI TTS API.
    """
    # Trim to 4000 characters (TTS limit)
    trimmed_text = text[:4000]
    
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=trimmed_text
        )
        return response.content
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")
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
