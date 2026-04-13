import os
import uuid
import logging

logger = logging.getLogger(__name__)

def transcribe_audio(openai_client, audio_bytes: bytes, file_format: str = "wav") -> str:
    """
    Transcribes audio bytes using OpenAI Whisper API.
    """
    temp_filename = f"temp_{uuid.uuid4()}.{file_format}"
    temp_path = os.path.join("temp", temp_filename)
    
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
            
        with open(temp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="text"
            )
        return transcript.strip()
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
