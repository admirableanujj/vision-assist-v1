from abc import ABC, abstractmethod

class VoiceEngineAC(ABC):
    """
    Abstract Component for VisionAssist Voice Operations.
    Ensures all derived audio modules implement standard execution flows.
    """

    @abstractmethod
    def initialize_engine(self) -> None:
        """Configure driver settings, microphone inputs, or API keys."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Main execution entry point for processing audio or text."""
        pass