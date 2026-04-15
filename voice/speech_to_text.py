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
            model="gemini-2.0-flash",
            contents=[
                "Please transcribe this audio exactly as heard. Do not add any extra text or descriptions.",
                {"mime_type": f"audio/{file_format}", "data": audio_bytes}
            ]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error transcribing with Gemini: {e}")
        return ""
