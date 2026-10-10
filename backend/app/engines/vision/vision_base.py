# app/vision_engine/vision_base.py
from abc import ABC, abstractmethod

class BaseVisionEngine(ABC):
    """
    Abstract Base Class for environmental scanner architectures.
    """
    @abstractmethod
    def scan_frame(self, image_buffer) -> dict:
        """
        Process a visual frame to extract detected items.

        Returns:
            dict: {
                "detections": list of {"label": str, "confidence": float, "box": tuple | None}
                    per item discovered in the frame, highest confidence first,
                "annotated_frame": the frame with detections visually marked, or
                    None if a rendered frame isn't available.
            }
        """
        pass  # pragma: no cover