# app/engines/ml/query_classifier.py
"""
Query Semantic Classifier Pipeline

Analyzes incoming natural language speech transcriptions to determine intents
like locate, general knowledge, note-taking, or system alarms.

Usage:
    from ml_engine.query_classifier import QueryClassifier
    intent_data = QueryClassifier(ml_brain.local_client).classify(text)

Dependencies:
    ollama>=0.6.2
    json
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"        
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import json
import logging
# from app.core.logging_config import get_logger

# logger = get_logger(__name__)
logger = logging.getLogger("query_classifier")

class QueryClassifier:
    def __init__(self, ollama_client, model_name: str = "llama3"):
        self.client = ollama_client
        self.model = model_name
        
    def classify(self, user_query: str) -> dict:
        """
        Runs semantic profiling on raw text inputs via local LLM.
        Returns a dictionary containing 'intent' and 'structured_payload'.
        """
        if not user_query:
            return {"intent": "general", "payload": ""}

        # System prompt forces the model to act as a strict classification router
        system_instructions = (
            "You are an AI intent classification router. You must categorize the user's input "
            "into exactly one of these intents: 'locate', 'note', 'alarm', or 'general'.\n"
            "Rules:\n"
            "- 'locate': Searching for missing physical belongings (e.g., keys, phone).\n"
            "- 'note': Requests to remember, log, write down, or recall a personal note.\n"
            "- 'alarm': Requests dealing with timers, alarms, reminders, or scheduling clock events.\n"
            "- 'general': General knowledge, conversation, weather, or anything else outside the scopes above.\n\n"
            "Return output strictly as a JSON object matching this schema:\n"
            '{"intent": "intent_string", "extracted_target": "extracted target item or text content"}'
        )

        try:
            response = self.client.generate(
                model=self.model,
                system=system_instructions,
                prompt=f"Classify this input query immediately: '{user_query}'",
                format="json", # Tells Ollama's backend to strictly force valid JSON formatting
                options={"temperature": 0.0} # Lower temp forces highly reproducible classifications
            )
            
            # Safely parse structural string tokens back into native dictionary items
            parsed_result = json.loads(response.get("response", "{}"))
            intent = parsed_result.get("intent", "general")
            payload = parsed_result.get("extracted_target", user_query)
            logger.info(f"Query classified as '{intent}' with payload: {str(payload)[:120]}")
            return {"intent": intent, "payload": payload}
        except Exception as e:
            logger.warning(f"Intent engine fallback triggered: {e}")
            return {"intent": "general", "payload": user_query}