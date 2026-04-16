import logging
import os

logger = logging.getLogger(__name__)

def transcribe_audio(audio_bytes: bytes, gemini_client, file_format: str = "wav") -> str:
    """
    Transcribes audio using Gemini's multimodal capabilities (completely free).
    """
    try:
        # Gemini can process audio directly as bytes
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {"text": "Transcribe this audio accurately. If it is a question about a document, emphasize the key keywords."},
                {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}}
            ]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error transcribing with Gemini: {e}")
        return ""
