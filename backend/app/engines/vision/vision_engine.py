# app/engines/vision/vision_engine.py
"""
YOLO / YOLOE object detection engine.

Usage:
    from app.engines.vision import VisionTracker
"""
__license__ = "MIT"
__version__ = "1.0.1"


import os
import random
from .vision_base import BaseVisionEngine
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Detections below this confidence score are discarded.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Nano variant for CPU-only Docker inference. Baked into the image at
# /opt/weights (see Dockerfile) — outside /workspace, which docker-compose's
# dev bind-mount would otherwise hide. Override via YOLO_MODEL_PATH.
DEFAULT_LOCAL_WEIGHTS = "/opt/weights/yolo26n.pt"

# Open-vocabulary variant — see YOLO_VS_YOLOE_GUIDE.md for when to use it.
DEFAULT_YOLOE_WEIGHTS = "/opt/weights/yoloe-26n-seg.pt"

# Standard 80-class MS-COCO label set. Passed to YOLOE's set_classes() so it
# detects at least as much as YOLOVisionEngine does — an open-vocabulary model
# only ever detects classes it was explicitly told about.
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Curated home-item vocabulary, filtered from the RAM++ tag list down to
# physical objects plausible indoors; a handful of gaps RAM++ doesn't cover
# (kettle, mop, bucket, doorbell, flashlight, wallet) added manually.
HOME_ITEM_CLASSES = [
    "air conditioner", "alarm clock", "apple", "apron", "armchair", "armoire", "backpack",
    "baking sheet", "banana", "bath", "bath towel", "bathroom cabinet", "bathroom mirror",
    "bathroom sink", "battery", "bed", "bed frame", "bedcover", "bedding", "bedside lamp",
    "belt", "bird cage", "blanket", "blender", "board game", "book", "bookcase", "bookshelf",
    "boot", "bottle", "bottle opener", "bowl", "bread", "briefcase", "broom", "bucket",
    "building block", "bulletin board", "butter", "cabinet", "calculator", "can",
    "can opener", "candle", "candle holder", "carpet", "carrot", "cat", "ceiling fan",
    "charger", "cheese", "chopstick", "cleaning product", "clock", "closet", "coat",
    "coffee machine", "coffeepot", "colander", "comb", "computer chair", "computer monitor",
    "cork", "couch", "crayon", "cream", "cup", "curtain", "cutting board", "desktop computer",
    "detergent", "dish washer", "dishes", "dishrag", "dog", "doll", "door handle", "doorbell",
    "doormat", "drawer", "dresser", "drum", "dustpan", "duvet", "earphone", "egg", "egg tart",
    "electric outlet", "extension cord", "face towel", "fan", "file cabinet", "fireplace",
    "first-aid kit", "fishbowl", "flashlight", "folder", "food processor", "fork", "fridge",
    "fruit", "frying pan", "game controller", "gas stove", "glove", "grape", "grater",
    "guitar", "hair drier", "hairbrush", "hamper", "hand towel", "handbag", "hanger", "hat",
    "houseplant", "induction cooker", "iron", "ironing board", "jacket", "jar", "juicer",
    "kettle", "keyboard", "keys", "kitchen cabinet", "kitchen counter", "kitchen hood",
    "kitchen island", "kitchen knife", "kitchen sink", "kitchen table", "kitchen utensil",
    "kitchen window", "kitchenware", "knife", "ladle", "lamp", "laptop", "laundry basket",
    "lemon", "light switch", "loveseat", "luggage", "magazine", "mattress", "measuring cup",
    "medicine", "microwave", "milk", "mirror", "mixer", "mixing bowl", "monitor", "mop",
    "mouse", "mousepad", "mug", "napkin", "nightstand", "notebook", "notepad", "office chair",
    "office desk", "onion", "orange", "oven", "pan", "pantry", "paper towel", "pen", "pencil",
    "pencil case", "person", "photo frame", "piano", "picture frame", "pillow", "plate",
    "playing card", "pot", "potato", "pressure cooker", "printer", "puzzle", "razor",
    "recycling bin", "remote", "rice cooker", "rolling pin", "sandal", "scale", "scarf",
    "scissors", "sewing machine", "shampoo", "shoe", "shower curtain", "shower head",
    "side table", "sink", "slipper", "slow cooker", "soap", "soap dispenser", "spatula",
    "speaker", "spice rack", "sponge", "spoon", "stapler", "stove", "strainer", "strawberry",
    "sunglasses", "table", "table lamp", "tape", "tea pot", "teddy", "television",
    "thermometer", "throw pillow", "toaster", "toilet bowl", "toilet paper", "tomato",
    "tongs", "toothbrush", "toothpaste", "towel bar", "toy", "toy car", "tray", "umbrella",
    "vacuum", "vase", "vegetable", "video game", "violin", "wall clock", "wallet",
    "washing machine", "waste container", "watch", "webcam", "whisk", "whiteboard",
    "wine glass", "wok",
]

# Default YOLOE vocabulary: COCO classes + the curated home-item list above.
DEFAULT_CUSTOM_CLASSES = sorted(set(COCO_80_CLASSES + HOME_ITEM_CLASSES))

# Which real engine to use — config, not code, same pattern as YOLO_MODEL_PATH.
DEFAULT_VISION_MODEL_TYPE = "yolo"


class FallbackVisionEngine(BaseVisionEngine):
    """
    Heuristic-based mock engine used when YOLO dependencies, 
    weights, or CUDA allocations fail to initialize.
    """
    def __init__(self):
        logger.warning("VisionAssist initialized FallbackVisionEngine. Running on Simulated Vision matrix.")
        self.simulated_pool = ["keys", "phone", "wallet", "sunglasses", "backpack"]

    def scan_frame(self, image_buffer) -> dict:
        """Returns 1-2 random items from the mock pool to keep the app working."""
        if image_buffer is None:
            return {"detections": [], "annotated_frame": None}

        detected_count = random.randint(1, 2)
        labels = random.sample(self.simulated_pool, k=detected_count)
        detections = [
            {"label": label, "confidence": round(random.uniform(0.5, 0.95), 2), "box": None}
            for label in labels
        ]
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return {"detections": detections, "annotated_frame": None}


try:
    import cv2
    import numpy as np
    from ultralytics import YOLO, YOLOE

    def _decode_frame(image_buffer):
        """
        Normalize a file-like object exposing .getvalue(), raw bytes, or an
        already-decoded np.ndarray into a BGR np.ndarray that ultralytics can
        run inference on. Shared by every real engine. /ws/vision passes raw
        bytes received straight off the WebSocket; the other branches exist
        for any other caller (tests, a future REST upload endpoint, etc.).
        """
        if isinstance(image_buffer, np.ndarray):
            return image_buffer
        if hasattr(image_buffer, "getvalue"):
            raw_bytes = image_buffer.getvalue()
        elif isinstance(image_buffer, (bytes, bytearray)):
            raw_bytes = image_buffer
        else:
            raw_bytes = image_buffer.read()
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _resolve_custom_classes():
        """VISION_CUSTOM_CLASSES='keys,wallet,...' overrides DEFAULT_CUSTOM_CLASSES for YOLOE."""
        raw = os.getenv("VISION_CUSTOM_CLASSES")
        if raw:
            return [c.strip() for c in raw.split(",") if c.strip()]
        return list(DEFAULT_CUSTOM_CLASSES)

    def _draw_detections(frame, detections):
        """
        Draws straight from `detections` rather than ultralytics' own
        Results.plot(), which renders differently per model task (YOLOE's
        segmentation checkpoints also draw masks) — this guarantees the image
        always matches what scan_frame() reports, for every engine.
        """
        annotated = frame.copy()
        for d in detections:
            if d["box"] is None:
                continue
            x1, y1, x2, y2 = (int(round(c)) for c in d["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{d['label']} {d['confidence']:.0%}"
            text_y = max(y1 - 8, 12)  # keep the label on-frame if the box starts near the top edge
            cv2.putText(
                annotated, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
            )
        return annotated

    class _UltralyticsScanMixin:
        """
        scan_frame() is identical across every ultralytics-backed engine (YOLO,
        YOLOE, or any future variant) — only model loading differs. Shared here
        so YOLOVisionEngine and YOLOEVisionEngine can't drift out of sync.
        Requires self.model, self.confidence_threshold, self._fallback.
        """
        def scan_frame(self, image_buffer) -> dict:
            if image_buffer is None:
                return {"detections": [], "annotated_frame": None}

            if self.model is None:
                return self._fallback.scan_frame(image_buffer)

            try:
                frame = _decode_frame(image_buffer)
                if frame is None:
                    return {"detections": [], "annotated_frame": None}

                import time as _time
                _start = _time.time()
                results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
                logger.info(f"Inference completed in {_time.time() - _start:.3f}s for {type(self).__name__}")

                detections = []
                for result in results:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence < self.confidence_threshold:
                            continue
                        class_id = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        label_name = result.names[class_id]
                        detections.append({
                            "label": label_name,
                            "confidence": round(confidence, 4),
                            "box": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                        })
                        logger.debug(f"Detected: {label_name} ({confidence:.2%})")
                detections.sort(key=lambda d: d["confidence"], reverse=True)

                # BGR (OpenCV's native order) to RGB for callers.
                annotated_frame = cv2.cvtColor(_draw_detections(frame, detections), cv2.COLOR_BGR2RGB)

                return {"detections": detections, "annotated_frame": annotated_frame}

            except Exception as e:
                logger.warning(f"{type(self).__name__} inference failed: {e!r}. Falling back to simulated detection.")
                return self._fallback.scan_frame(image_buffer)

    class YOLOVisionEngine(_UltralyticsScanMixin, BaseVisionEngine):
        def __init__(self, weights_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
            # FINE_TUNED_YOLO_MODEL_PATH takes priority over the shared
            # YOLO_MODEL_PATH (kept for back-compat) specifically so pointing
            # a fine-tuned checkpoint at YOLOVisionEngine can never also get
            # picked up by YOLOEVisionEngine below — both engines can be
            # resident in the same process (get_engine() caches each), and a
            # plain-YOLO checkpoint loaded through the YOLOE class fails
            # outright (wrong architecture), silently degrading YOLOE to mock
            # detections. `or` (not os.getenv's default arg) so an
            # empty-string env var — what Compose passes when unset — never
            # wins over the real default.
            self.weights_path = (
                weights_path
                or os.getenv("FINE_TUNED_YOLO_MODEL_PATH")
                or os.getenv("YOLO_MODEL_PATH")
                or DEFAULT_LOCAL_WEIGHTS
            )
            self.confidence_threshold = confidence_threshold
            self._fallback = FallbackVisionEngine()
            try:
                self.model = YOLO(self.weights_path)
                logger.info(f"YOLOVisionEngine loaded '{self.weights_path}' (conf>={confidence_threshold}).")
            except Exception as e:
                logger.warning(
                    f"Failed to load YOLO weights '{self.weights_path}': {e!r}. "
                    f"YOLOVisionEngine will use simulated detections until this is fixed.")
                self.model = None

    class YOLOEVisionEngine(_UltralyticsScanMixin, BaseVisionEngine):
        """
        Open-vocabulary variant — detects classes named via set_classes() instead
        of a fixed pretrained label set. Same scan_frame() contract and closed-set
        fallback behavior as YOLOVisionEngine; only model loading differs.
        """
        def __init__(self, weights_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, classes=None):
            # Deliberately does NOT read FINE_TUNED_YOLO_MODEL_PATH — that
            # var is YOLOVisionEngine-only, by design (see its comment).
            # Same empty-string guard as YOLOVisionEngine.
            self.weights_path = weights_path or os.getenv("YOLO_MODEL_PATH") or DEFAULT_YOLOE_WEIGHTS
            self.confidence_threshold = confidence_threshold
            self.classes = classes or _resolve_custom_classes()
            self._fallback = FallbackVisionEngine()
            try:
                self.model = YOLOE(self.weights_path)
                self.model.set_classes(self.classes)
                logger.info(
                    f"YOLOEVisionEngine loaded '{self.weights_path}' with classes={self.classes} (conf>={confidence_threshold}).")
            except Exception as e:
                logger.warning(
                    f"Failed to load YOLOE weights '{self.weights_path}': {e!r}. "
                    f"YOLOEVisionEngine will use simulated detections until this is fixed.")
                self.model = None

    _ENGINE_TYPES = {"yolo": YOLOVisionEngine, "yoloe": YOLOEVisionEngine}
    _selected_type = os.getenv("VISION_MODEL_TYPE", DEFAULT_VISION_MODEL_TYPE).strip().lower()
    VisionTracker = _ENGINE_TYPES.get(_selected_type, YOLOVisionEngine)

    # Per-engine instance cache so callers (e.g. a per-connection engine
    # choice over /ws/vision) can hold both YOLO and YOLOE resident at once
    # and switch between them without a restart. Lazy: each engine is only
    # constructed (and pays its load cost) the first time it's actually
    # requested, not eagerly for both at import time.
    _engine_cache: dict = {}

    def get_engine(engine_type: str = None):
        key = (engine_type or _selected_type or DEFAULT_VISION_MODEL_TYPE).strip().lower()
        if key not in _engine_cache:
            cls = _ENGINE_TYPES.get(key, YOLOVisionEngine)
            _engine_cache[key] = cls()
        return _engine_cache[key]

except ImportError as e:
    logger.critical(f"Vision dependencies unavailable: {e}")
    VisionTracker = FallbackVisionEngine

    def get_engine(engine_type: str = None):
        return FallbackVisionEngine()