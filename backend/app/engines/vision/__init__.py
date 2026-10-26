# app/engines/vision/__init__.py
from .vision_engine import VisionTracker, FallbackVisionEngine, get_engine

__all__ = ["VisionTracker", "FallbackVisionEngine", "get_engine"]

# YOLOVisionEngine/YOLOEVisionEngine only exist if cv2/numpy/ultralytics loaded
# successfully (see the try/except ImportError boundary in vision_engine.py) —
# exposed here too, in addition to VisionTracker, so callers that need a
# specific engine explicitly, rather than whatever VISION_MODEL_TYPE selected,
# can import them from the package directly instead of reaching into the submodule.
try:
    from .vision_engine import YOLOVisionEngine, YOLOEVisionEngine
    __all__ += ["YOLOVisionEngine", "YOLOEVisionEngine"]
except ImportError:
    pass