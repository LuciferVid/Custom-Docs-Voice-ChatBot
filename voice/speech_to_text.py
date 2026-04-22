import logging
import os
import io

logger = logging.getLogger(__name__)

def transcribe_audio(audio_bytes: bytes, gemini_client, file_format: str = "wav") -> str:
    """
    Transcribes audio using Gemini 2.0 Flash's multimodal capabilities.
    """
    try:
        mime_type = f"audio/{file_format}"
        if file_format == "wav":
            mime_type = "audio/wav"
        elif file_format == "mp3":
            mime_type = "audio/mpeg"

        # Try 2.0 first, then fallback
        model_id = "gemini-2.0-flash"
        try:
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=[
                    "Transcribe the following audio content into text. Return ONLY the transcription, nothing else.",
                    {"mime_type": mime_type, "data": audio_bytes}
                ]
            )
        except:
            model_id = "gemini-1.5-flash-latest"
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=[
                    "Transcribe the following audio content into text. Return ONLY the transcription, nothing else.",
                    {"mime_type": mime_type, "data": audio_bytes}
                ]
            )
        
        transcription = response.text.strip()
        logger.info(f"Gemini {model_id} Transcription: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing with Gemini: {e}")
        return ""


