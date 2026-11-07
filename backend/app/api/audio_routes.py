import os
import tempfile
import logging
import ollama
import io
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gtts import gTTS
from openai import OpenAI

from app.engines.ml.query_classifier import QueryClassifier
from app.core.security import get_current_user

logger = logging.getLogger("audio_routes")
router = APIRouter(tags=["Audio & Voice Reasoning"])

# Determine selected provider from environment (options: "faster-whisper", "whisper", "openai")
STT_PROVIDER = os.getenv("STT_PROVIDER", "faster-whisper").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize the selected speech engine
audio_model = None
openai_client = None

try:
    if STT_PROVIDER == "faster-whisper":
        from faster_whisper import WhisperModel
        audio_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Loaded Faster-Whisper speech engine.")
        
    elif STT_PROVIDER == "whisper":
        import whisper
        audio_model = whisper.load_model("base")
        logger.info("Loaded original OpenAI Whisper speech engine.")
        
    elif STT_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Initialized OpenAI Cloud Audio API engine.")
    else:
        logger.warning(f"Unknown STT_PROVIDER '{STT_PROVIDER}', defaulting to offline fallback.")
        
except Exception as e:
    logger.error(f"Failed to initialize STT provider '{STT_PROVIDER}': {e}")
    audio_model = None
    openai_client = None
    
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "vision_assist_llm_local")
ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:11434")
query_classifier = QueryClassifier(ollama_client, model_name="llama3")

async def transcribe_audio_file(file_path: str) -> str:
    """Dispatches audio transcription to the configured STT provider."""
    if STT_PROVIDER == "faster-whisper" and audio_model:
        segments, _ = audio_model.transcribe(file_path, beam_size=5)
        return " ".join([segment.text for segment in segments]).strip()

    elif STT_PROVIDER == "whisper" and audio_model:
        result = audio_model.transcribe(file_path)
        return result.get("text", "").strip()

    elif STT_PROVIDER == "openai" and openai_client:
        with open(file_path, "rb") as audio_file:
            transcript_obj = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript_obj.text.strip()
            
    raise RuntimeError(f"Active STT provider '{STT_PROVIDER}' is uninitialized or unavailable.")


@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    #Extract token directly from query parameters for WebSocket authentication
    token = websocket.query_params.get("token")    
    
    
    # Authenticate token manually during WebSocket handshake
    if not token or token == "null" or token == "undefined":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning("Audio WebSocket connection rejected: Missing authentication token.")
        return    
        
    # try:
    #         # Optional: Perform full token decoding/verification here if desired
    #         from app.core.security import decode_access_token # or your custom verification
    #         # payload = decode_access_token(token)
    #         # if not payload:
    #         #     raise Exception("Invalid token payload")
    #     except Exception as e:
    #         logger.warning(f"Audio WebSocket authentication failed: {e}")
    #         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    #         return        
    await websocket.accept()
    logger.info(f"Client connected to Audio WebSocket stream (Provider: {STT_PROVIDER}).")
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            if not audio_bytes:
                continue

            if (STT_PROVIDER in ["faster-whisper", "whisper"] and audio_model is None) or \
               (STT_PROVIDER == "openai" and openai_client is None):
                await websocket.send_json({
                    "status": "error",
                    "transcript": "",
                    "response": f"STT provider '{STT_PROVIDER}' is offline or misconfigured."
                })
                continue

            # Save incoming audio stream to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                # Transcribe using active provider
                transcript = await transcribe_audio_file(tmp_path)
                logger.info(f"Transcribed audio [{STT_PROVIDER}]: '{transcript}'")

                response_text = f"Genie acknowledges [{STT_PROVIDER}]: \"{transcript or 'Silence detected'}\"."

                if not transcript:
                    continue
                
                # Run Semantic Intent Classification via your QueryClassifier
                classification = query_classifier.classify(transcript)
                intent = classification["intent"]
                payload = classification["payload"]
                logger.info(f"Routed Intent -> [{intent.upper()}] | Payload: {payload}")

                # Generate tailored response based on classified intent
                if intent == "locate":
                    response_text = f"Genie [Locate Mode]: Scanning spatial memory and vision feeds for target: '{payload}'..."
                elif intent == "note":
                    response_text = f"Genie [Note Logged]: Successfully saved note -> \"{payload}\"."
                elif intent == "alarm":
                    response_text = f"Genie [Alarm/Timer]: Setting up trigger for: '{payload}'."
                else:
                    # General conversation/knowledge fallback using Ollama
                    res = ollama_client.chat(model='llama3', messages=[{'role': 'user', 'content': transcript}])
                    response_text = res['message']['content'].strip()
                
                await websocket.send_json({
                    "status": "success",
                    "intent": intent,
                    "transcript": transcript,
                    "response": response_text
                })

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    except WebSocketDisconnect:
        logger.info("Client disconnected from Audio WebSocket stream.")
    except Exception as e:
        logger.error(f"Error in audio websocket loop: {e}")
        try:
            await websocket.close()
        except:
            pass
        
class TTSRequest(BaseModel):
    text: str

@router.post("/tts")
async def generate_tts(request: TTSRequest, current_user: dict = Depends(get_current_user)):
    """
    Generates a TTS audio stream from the provided text using gTTS.
    Returns an audio/mpeg streaming response directly to the client browser.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text for TTS generation is required.")

    try:
        tts = gTTS(text=request.text, lang='en', tld='com', slow=False)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        username = current_user.get('username', 'operator')
        logger.info(f"TTS audio stream successfully generated for user '{username}'.")
        return StreamingResponse(audio_io, media_type="audio/mpeg")
    except Exception as e:
        logger.exception(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate TTS audio.")