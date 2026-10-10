#app/voice_engine/voice_tts.py
"""
Browser-Side Text-To-Speech (TTS) Engine

Converts model response text strings into streaming audio assets using gTTS.
This bypasses headless Linux container audio card errors by handing off audio playback 
directly to the client's local browser window.

Usage:
    Imported dynamically into the core framework via:
    from voice_engine.voice_tts import TextToSpeechConverter

Dependencies:
    gTTS==2.5.1
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"       
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import os
import sys
from .voice_base import VoiceEngineAC
from gtts import gTTS
from logging_config import get_logger
import logging

_base_logger = get_logger(__name__)


def _make_adapter(base: logging.Logger, extra: dict | None = None) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(base, extra or {"component": "voice_tts"})

class TextToSpeechConverter(VoiceEngineAC):
    """
    Converts string responses into auditory files suitable for web UI streaming,
    bypassing headless container sound card limitations.
    """
    def __init__(self, speech_rate: int = 165):
        self.speech_rate = speech_rate
        self.output_filename = "response_vocal.mp3"
        # Create a LoggerAdapter to carry repeated context (e.g., user_id)
        self.logger = _make_adapter(_base_logger)
        self.initialize_engine()

    def initialize_engine(self) -> None:
        self.logger.info("TextToSpeech pipeline configured for browser-side streaming delivery.", extra={"speech_rate": self.speech_rate})

    def execute(self, response_text: str) -> str:
        """
        Generates an audio file from text.
        Returns the string path to the audio file if successful, or an empty string.
        """
        if not response_text:
            return ""

        try:
            # Clean up old audio files before generating a new one
            if os.path.exists(self.output_filename):
                os.remove(self.output_filename)

            # Use edge-tts or gtts via pip to create file buffers without a sound card
            tts = gTTS(text=response_text, lang='en', tld='com')
            tts.save(self.output_filename)
            
            self.logger.info("Generated TTS audio file", extra={"path": self.output_filename})
            return self.output_filename
        except Exception as e:
            self.logger.exception(f"Audio file generation failure: {e}", extra={"response_text": response_text, "speech_rate": self.speech_rate})
            return ""