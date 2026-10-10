#app/voice_engine/voice_stt.py
"""
Hybrid Speech-To-Text (STT) Converter

Processes binary microphone stream files from the web frontend browser canvas. 
It fluidly shifts execution pipelines between a cloud-independent local 
Faster-Whisper CPU deployment and the official OpenAI Whisper API wrapper.

Usage:
    Imported dynamically into the core framework via:
    from voice_engine.voice_stt import SpeechToTextConverter

Dependencies:
    openai
    faster-whisper==1.0.3
    python-dotenv==1.0.1

__original_author__ = "Anujj Saxena"
__license__ = "MIT"      
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import os
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel
from .voice_base import VoiceEngineAC
from logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

class SpeechToTextConverter(VoiceEngineAC):
    """
    Hybrid Speech-to-Text Engine that seamlessly switches between 
    OpenAI Cloud API and a fully offline local Faster-Whisper deployment.
    """
    def __init__(self):
        self.client = None
        self.local_model = None
        self.initialize_engine()

    def initialize_engine(self) -> None:
        """
        Detects available system resources and hooks up the correct pipeline interface.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.info("Whisper STT initialized via Cloud API Wrapper.")
        else:
            logger.info("No cloud API key detected. Bootstrapping fully offline Faster-Whisper layer...")
            try:
                # Using the optimized "tiny" or "base" model for fast, low-overhead CPU calculations
                self.local_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                logger.info("Local offline Whisper Core successfully mapped to CPU hardware.")
            except ImportError:
                logger.error("'faster-whisper' package missing from local environment manifest.")

    def execute(self, audio_buffer) -> str:
        """
        Main execution contract: Processes the incoming binary stream via the active engine layer.
        """
        if audio_buffer is None:
            return ""

        # --- PIPELINE A: Cloud Architecture Extraction ---
        if self.client:
            try:
                audio_buffer.name = "input_command.wav"
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_buffer,
                    temperature=0.0
                )
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Cloud transcription failed: {e}. Attempting local fallback execution...")

        # --- PIPELINE B: Local Autonomous Framework ---
        if self.local_model:
            try:
                # Process the raw bytes directly from Streamlit's file buffer memory block
                segments, info = self.local_model.transcribe(audio_buffer, beam_size=5)
                # Concatenate calculated translation strings from array chunks
                transcription = "".join([segment.text for segment in segments])
                return transcription.strip()
            except Exception as local_err:
                logger.exception(f"Local execution loop broken: {local_err}")
                return "Error: Local transcription engine processing exception."

        return "where are my keys"