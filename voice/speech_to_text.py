from google import genai
import logging
import uuid
import os

logger = logging.getLogger(__name__)

def transcribe_audio(openai_client, audio_bytes: bytes, gemini_client=None, file_format: str = "wav") -> str:
    """
    Transcribes audio. Defaults to Gemini if client is provided, falls back to OpenAI Whisper.
    """
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # If Gemini is provided, use it (Free/Multimodal)
    if gemini_client:
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
            # Fallback will continue below

    # Fallback to OpenAI Whisper (Requires paid key)
    temp_filename = f"temp_{uuid.uuid4()}.{file_format}"
    temp_path = os.path.join("temp", temp_filename)
    
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
        logger.error(f"Error transcribing with OpenAI: {e}")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
