import logging
import os
import io

logger = logging.getLogger(__name__)

def transcribe_audio(audio_bytes: bytes, groq_client, file_format: str = "wav") -> str:
    """
    Transcribes audio using Groq's blazing fast Whisper-Large-v3-Turbo model.
    """
    try:
        # Wrap bytes in a file-like object with a name so Groq accepts it
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{file_format}"
        audio_file.seek(0) # Ensure we are at the start
        
        response = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text"
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Error transcribing with Groq: {e}")
        return ""
