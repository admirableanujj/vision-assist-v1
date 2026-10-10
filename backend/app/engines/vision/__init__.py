# app/vision_engine/__init__.py
from .vision_engine import VisionTracker, FallbackVisionEngine

__all__ = ["VisionTracker", "FallbackVisionEngine"]

# YOLOVisionEngine/YOLOEVisionEngine only exist if cv2/numpy/ultralytics loaded
# successfully (see the try/except ImportError boundary in vision_engine.py) —
# exposed here too, in addition to VisionTracker, so callers that need to pick
# a specific engine explicitly (e.g. app.py's model selector) can import them
# from the package directly rather than reaching into the submodule.
try:
    from .vision_engine import YOLOVisionEngine, YOLOEVisionEngine
    __all__ += ["YOLOVisionEngine", "YOLOEVisionEngine"]
except ImportError:
    pass