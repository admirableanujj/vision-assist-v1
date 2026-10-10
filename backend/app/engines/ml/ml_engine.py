# app/ml_engine/ml_engine.py
"""
Ollama and LangChain Hybrid ML Engine

This module interfaces directly with the local Ollama SDK daemon to coordinate 
LLM reasoning patterns. It features a reliable cloud fallback strategy via 
LangChain OpenAI clients if local assets go offline.

Usage:
    Imported dynamically into the Streamlit app execution tree via:
    from ml_engine.ml_engine import OllamaMLEngine

Dependencies:
    ollama
    langchain
    langchain-openai
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"    
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import os
import ollama
from dotenv import load_dotenv
from .ml_base_engine import BaseMLEngine
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Load local environment variables (for OpenAI API keys, etc.)
load_dotenv()

# class OllamaMLEngine(BaseMLEngine):
#     """
#     Enhanced concrete implementation of BaseMLEngine natively utilizing the Ollama SDK 
#     with an integrated LangChain + OpenAI cloud fallback framework for voice classification.
#     """
#     def __init__(self, host: str = None, default_model: str = "llama3"):
#         # Prefer the OLLAMA_HOST env var (set by docker-compose) so the host is
#         # configurable without touching code. Fall back to the container default.
#         self.host = host or os.getenv("OLLAMA_HOST", "http://vision_assist_llm_local:11434")
#         self.model = default_model
        
#         # Configure client bindings for the official Ollama SDK
#         self.local_client = ollama.Client(host=self.host)
#         print(f"[INFO] OllamaMLEngine natively bound over SDK client via {self.host}")
        
#         # Optional LangChain / OpenAI Fallback Integration
#         self.fallback_llm = None
#         self.fallback_enabled = False
#         if os.getenv("OPENAI_API_KEY") is not None:
#             try:
#                 from langchain_openai import ChatOpenAI
#                 # Initialize a highly concise cloud fallback engine for latency insurance
#                 self.fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
#                 self.fallback_enabled = True
#                 print("[INFO] LangChain OpenAI cloud hybrid fallback engine securely configured.")
#             except Exception as e:
#                 # Keep fallback_enabled/fallback_llm in sync so callers never see
#                 # fallback_enabled=True with a missing fallback_llm.
#                 print(f"[WARN] LangChain OpenAI fallback failed to initialize: {e}. Continuing with local Ollama only.")

#     # def tokenize_text(self, input_text: str) -> list:
#     #     """
#     #     Natively pulls token index maps via the local Ollama SDK client wrapper.
#     #     """
#     #     try:
#     #         response = self.local_client.tokenize(model=self.model, prompt=input_text)
#     #         return response.get("tokens", [])
#     #     except Exception as e:
#     #         print(f"[WARN] Local Tokenizer failed: {e}. Falling back to empty map.")
#     #         return []

#     def tokenize_text(self, input_text: str) -> list:
#             """
#             Natively pulls token index maps via the local Ollama SDK client wrapper.
#             """
#             if not input_text:
#                 return []
                
#             try:
#                 # The correct client layout uses the embedded .embed or sub-client structures
#                 # For direct raw string tokenization arrays:
#                 response = self.local_client.embed(model=self.model, input=input_text)
                
#                 # If your downstream layout explicitly needs integer index locations:
#                 # We can map the embedded float arrays or query the internal model tokens:
#                 return response.get("embeddings", [[]])[0][:12] # Slices out first 12 structural markers
                
#             except Exception as e:
#                 # Fallback routine so the Streamlit UI never completely freezes up
#                 print(f"[WARN] Local Tokenizer failed: {e}. Falling back to clean word-split token weights.")
#                 return [len(word) for word in input_text.split()]

#     def tokenize_text(self, input_text: str) -> list:
#         """
#         Simulates token indexes securely without requiring an external 
#         Ollama embedding model infrastructure setup.
#         """
#         if not input_text:
#             return []
#         try:
#             # Converts letters directly into fake token IDs for UI display metrics
#             return [ord(char) % 100 for char in input_text.split()[:12]]
#         except Exception:
#             return []

    
#     def generate_response(self, user_text: str, tracking_context: str) -> str:
#         """
#         Executes structural query evaluation and context response engineering.
#         """
#         system_prompt = (
#             "You are VisionCore-ML, a highly precise assistive voice system.\n"
#             "Analyze the voice command and answer accurately using ONLY the tracking context provided.\n"
#             f"Active Tracking Context Matrix: {tracking_context}\n"
#             f"User Voice Request: {user_text}\n"
#             "Keep your final spoken output clear, conversational, and under two sentences."
#         )

#         try:
#             # 1. Attempt Native Local Inference via Ollama SDK
#             response = self.local_client.generate(
#                 model=self.model,
#                 prompt=system_prompt,
#                 options={"temperature": 0.2, "stop": ["\n\n"]}
#             )
#             return response.get("response", "").strip()

#         except Exception as local_error:
#             print(f"[SYSTEM WARN] Local ML Engine generation failed: {local_error}")
            
#             # 2. Trigger Intelligent Cloud Failover if Local Container is Offline/Stalled
#             if self.fallback_enabled:
#                 print("[SYSTEM INFO] Engaging LangChain cloud backup pipelines...")
#                 try:
#                     from langchain_core.messages import SystemMessage, HumanMessage
#                     messages = [
#                         SystemMessage(content="You are VisionCore-ML fallback system. Answer concisely based on given inventory state."),
#                         HumanMessage(content=f"Context: {tracking_context}\nQuery: {user_text}")
#                     ]
#                     cloud_response = self.fallback_llm.invoke(messages)
#                     return cloud_response.content.strip()
#                 except Exception as cloud_error:
#                     return f"System Orchestration Failure. Cloud chain also unreachable: {cloud_error}"
            
#             return "VisionAssist Core is currently recovering or adjusting container system links."

#     def generate_general_response(self, user_text: str) -> str:
#         """
#         Handles general knowledge queries outside the lost-item domain.
#         Primary path: OpenAI gpt-4o-mini via LangChain (requires OPENAI_API_KEY).
#         Fallback path: local Ollama inference when no API key is present.
#         """
#         system_prompt = (
#             "You are VisionAssist, a helpful voice assistant. "
#             "Answer the user's question clearly and concisely in one or two sentences. "
#             "Do not reference lost items or object tracking unless the user asks about them."
#         )

#         # Path A: Cloud LLM via LangChain (OpenAI key present)
#         if self.fallback_enabled:
#             try:
#                 from langchain_core.messages import SystemMessage, HumanMessage
#                 messages = [
#                     SystemMessage(content=system_prompt),
#                     HumanMessage(content=user_text),
#                 ]
#                 cloud_response = self.fallback_llm.invoke(messages)
#                 return cloud_response.content.strip()
#             except Exception as cloud_err:
#                 print(f"[WARN] Cloud LLM failed: {cloud_err}. Falling back to local Ollama...")

#         # Path B: Local Ollama (no API key, or cloud failed)
#         try:
#             response = self.local_client.generate(
#                 model=self.model,
#                 prompt=f"{system_prompt}\n\nUser: {user_text}\nAssistant:",
#                 options={"temperature": 0.4, "stop": ["\n\n"]},
#             )
#             return response.get("response", "").strip()
#         except Exception as local_err:
#             print(f"[ERROR] Local Ollama general response failed: {local_err}")
#             return "I'm unable to answer right now. Please check that the Ollama container is running."

class OllamaMLEngine(BaseMLEngine):
    """
    Enhanced concrete implementation of BaseMLEngine natively utilizing the Ollama SDK 
    with an integrated LangChain + OpenAI cloud fallback framework for voice classification.
    """
    def __init__(self, host: str = None, default_model: str = "llama3"):
        self.host = host or os.getenv("OLLAMA_HOST", "http://vision_assist_llm_local:11434")
        self.model = default_model
        
        # Configure client bindings for the official Ollama SDK
        self.local_client = ollama.Client(host=self.host)
        logger.info(f"OllamaMLEngine natively bound over SDK client via {self.host}")
        
        # Configure LangChain / OpenAI Fallback Integration
        self.fallback_llm = None
        self.fallback_enabled = False
        
        # Explicitly validate key to ignore empty strings or default placeholders (e.g., "sk-...")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key and not api_key.startswith("sk-..."):
            try:
                from langchain_openai import ChatOpenAI
                self.fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
                self.fallback_enabled = True
                logger.info("LangChain OpenAI cloud hybrid fallback engine securely configured.")
            except Exception as e:
                logger.warning(f"LangChain OpenAI fallback failed to initialize: {e}. Continuing with local Ollama only.")
        else:
            logger.info("Missing or placeholder OpenAI key. Cloud fallback disabled. Using 100% local Ollama.")

    def tokenize_text(self, input_text: str) -> list:
        """
        Simulates token indexes securely without requiring an external 
        Ollama embedding model infrastructure setup.
        """
        if not input_text:
            return []
        try:
            # Converts letters directly into fake token IDs for UI display metrics
            return [ord(char) % 100 for char in input_text.split()[:12]]
        except Exception:
            return []

    def generate_response(self, user_text: str, tracking_context: str) -> str:
        """
        Executes structural query evaluation and context response engineering.
        """
        system_prompt = (
            "You are VisionCore-ML, a highly precise assistive voice system.\n"
            "Analyze the voice command and answer accurately using ONLY the tracking context provided.\n"
            f"Active Tracking Context Matrix: {tracking_context}\n"
            f"User Voice Request: {user_text}\n"
            "Keep your final spoken output clear, conversational, and under two sentences."
        )

        import time as _time
        logger.info(f"[locate] Generating response for user_text='{user_text[:80]}'")
        try:
            # 1. Attempt Native Local Inference via Ollama SDK
            response = self.local_client.generate(
                model=self.model,
                prompt=system_prompt,
                options={"temperature": 0.2, "stop": ["\n\n"]}
            )
            _resp = response.get("response", "").strip()
            logger.info(f"[locate] Local response generated (len={len(_resp)})")
            return _resp

        except Exception as local_error:
            logger.warning(f"Local ML Engine generation failed: {local_error}")
            
            # 2. Trigger Intelligent Cloud Failover if Local Container is Offline/Stalled
            if self.fallback_enabled:
                logger.info("Engaging LangChain cloud backup pipelines...")
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content="You are VisionCore-ML fallback system. Answer concisely based on given inventory state."),
                        HumanMessage(content=f"Context: {tracking_context}\nQuery: {user_text}")
                    ]
                    cloud_response = self.fallback_llm.invoke(messages)
                    return cloud_response.content.strip()
                except Exception as cloud_error:
                    logger.exception(f"Cloud fallback invocation failed: {cloud_error}")
                    return f"System Orchestration Failure. Cloud chain also unreachable: {cloud_error}"
            
            return "VisionCore-ML is currently recovering or adjusting container system links."

    def generate_general_response(self, user_text: str) -> str:
        """
        Handles general knowledge queries outside the lost-item domain.
        Primary path: OpenAI gpt-4o-mini via LangChain (requires OPENAI_API_KEY).
        Fallback path: local Ollama inference when no API key is present.
        """
        system_prompt = (
            "You are VisionAssist, a helpful voice assistant. "
            "Answer the user's question clearly and concisely in one or two sentences. "
            "Do not reference lost items or object tracking unless the user asks about them."
        )

        import time as _time
        logger.info(f"[general] Generating general response for user_text='{user_text[:80]}'")
        # Path A: Cloud LLM via LangChain (OpenAI key present)
        if self.fallback_enabled:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_text),
                ]
                cloud_response = self.fallback_llm.invoke(messages)
                return cloud_response.content.strip()
            except Exception as cloud_err:
                logger.warning(f"Cloud LLM failed: {cloud_err}. Falling back to local Ollama...")

        # Path B: Local Ollama (no API key, or cloud failed)
        try:
            response = self.local_client.generate(
                model=self.model,
                prompt=f"{system_prompt}\n\nUser: {user_text}\nAssistant:",
                options={"temperature": 0.4, "stop": ["\n\n"]},
            )
            return response.get("response", "").strip()
        except Exception as local_err:
            logger.exception(f"Local Ollama general response failed: {local_err}")
            return "I'm unable to answer right now. Please check that the Ollama container is running."