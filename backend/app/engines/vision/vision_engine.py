# app/vision_engine/vision_engine.py
"""
YOLO Real-Time Object Tracking Pipeline

Loads optimized ultralytics computer vision weights inside the container framework 
to scan camera inputs, detect registered belongings, and update the session status log.

Usage:
    Imported dynamically into the core framework via:
    from vision_engine.vision_tracker import VisionTracker

Dependencies:
    ultralytics==8.2.0
    torch
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"      
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"


import os
import random
from .vision_base import BaseVisionEngine
from logging_config import get_logger

logger = get_logger(__name__)

# Detections below this confidence score are discarded ("eliminate outlying" detections).
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Local default: smallest/fastest variant for CPU-only Docker inference.
# YOLO26n over YOLO11n: fewer params (2.4M vs 2.6M), higher mAP (40.9 vs 39.5),
# and ~30% faster CPU inference (38.9ms vs 56.1ms) at the same size class — see
# the Milestone 4 model-selection writeup for the full comparison. Override
# per-environment via YOLO_MODEL_PATH (e.g. a larger GPU-appropriate weights
# file in a cloud deployment) without touching this code.
DEFAULT_LOCAL_WEIGHTS = "yolo26n.pt"

# YOLOE: Ultralytics' open-vocabulary variant. Detects arbitrary text-prompted
# classes (e.g. "keys"/"wallet"/"sunglasses", none of which are COCO classes
# YOLOVisionEngine can ever learn without fine-tuning) at the cost of lower
# per-class accuracy than a closed-set model — see YOLO_VS_YOLOE_GUIDE.md.
DEFAULT_YOLOE_WEIGHTS = "yoloe-26n-seg.pt"

# Standard 80-class MS-COCO label set, in the same order every COCO-pretrained
# YOLO checkpoint uses. Passed to YOLOE's set_classes() so it detects at least
# as much as YOLOVisionEngine does (person, cell phone, backpack, ...) — an
# open-vocabulary model only ever detects classes it was explicitly told about,
# so leaving this out (as an earlier version of this file did) means it can
# structurally never see a person in frame, no matter how confident it'd be.
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

# Curated "things found in a home" vocabulary — kitchen, living room, bedroom,
# bathroom, and workspace items. Sourced from the RAM++ tag list (Recognize
# Anything Model Plus, 4,585 classes — the same vocabulary YOLOE's "prompt-free"
# checkpoints ship with internally), filtered down to physical objects plausibly
# found indoors at home. Every entry was verified present in that source list
# except a handful of genuine gaps (kettle, mop, bucket, doorbell, flashlight,
# wallet — none of which RAM++ includes at all) added manually. Deliberately
# excludes RAM++'s wild animals, vehicles, outdoor scenery, buildings,
# professions, and abstract/verb tags (e.g. "aerobics", "action film") — those
# aren't home objects and would only dilute confidence across more candidates
# for no benefit. See YOLO_VS_YOLOE_GUIDE.md for the accuracy tradeoff of a
# larger class list, and FINE_TUNING_YOLO_GUIDE.md for why RAM++ was chosen as
# the source rather than hand-brainstorming a list from scratch.
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

# Classes YOLOEVisionEngine is prompted to detect by default: every COCO class
# (parity with YOLOVisionEngine) plus the curated home-item vocabulary above —
# a genuinely broad "whatever's in a home" list, not just the 3 items that
# originally motivated using YOLOE at all.
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
        # Common household objects to mock local identification loops
        self.simulated_pool = ["keys", "phone", "wallet", "sunglasses", "backpack"]

    def scan_frame(self, image_buffer) -> dict:
        """
        Simulates scanning a camera buffer by returning 1-2 random items
        from the tracking pool to keep the app working.
        """
        if image_buffer is None:
            return {"detections": [], "annotated_frame": None}

        # Simulate an automated detection confidence threshold pass
        detected_count = random.randint(1, 2)
        labels = random.sample(self.simulated_pool, k=detected_count)
        detections = [
            {"label": label, "confidence": round(random.uniform(0.5, 0.95), 2), "box": None}
            for label in labels
        ]
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        # No real image to draw on in the mock path — annotated_frame stays None.
        return {"detections": detections, "annotated_frame": None}


# --- REAL YOLO SYSTEM BOUNDARY ---
try:
    # Attempt to load your real computer vision stack
    import cv2
    import numpy as np
    from ultralytics import YOLO, YOLOE

    def _decode_frame(image_buffer):
        """
        Normalize a Streamlit camera_input buffer (bytes-like UploadedFile),
        raw bytes, or an already-decoded np.ndarray into a BGR np.ndarray
        that ultralytics can run inference on. Shared by every real engine.
        """
        if isinstance(image_buffer, np.ndarray):
            return image_buffer
        if hasattr(image_buffer, "getvalue"):
            raw_bytes = image_buffer.getvalue()  # st.camera_input()'s UploadedFile
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
        Draw a box + label + confidence for each detection directly from our
        own already-validated detection data, onto a copy of the decoded frame.

        Deliberately NOT using ultralytics' Results.plot() here — its default
        rendering varies by model task (segmentation checkpoints like YOLOE's
        also render mask overlays, keypoints, etc., depending on what the
        Results object carries), which is one more thing that can silently
        differ or fail between YOLO and YOLOE without changing what scan_frame()
        reports as detected. Drawing straight from `detections` guarantees the
        image always shows exactly what the text output says, for every engine.
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

                # conf= enforces the confidence cutoff at the ultralytics/NMS level.
                import time as _time
                _start = _time.time()
                results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
                _elapsed = _time.time() - _start
                logger.info(f"Inference completed in {_elapsed:.3f}s for {type(self).__name__}")

                detections = []
                for result in results:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        # Defensive re-check: guarantees "eliminate outlying"
                        # detections regardless of ultralytics' internal conf= behavior.
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

                # Draw straight from `detections` (see _draw_detections' docstring
                # for why, not ultralytics' Results.plot()), then convert BGR
                # (OpenCV's native order) to RGB so callers (e.g. Streamlit's
                # st.image) don't need to know this engine's internal color convention.
                annotated_frame = cv2.cvtColor(_draw_detections(frame, detections), cv2.COLOR_BGR2RGB)

                return {"detections": detections, "annotated_frame": annotated_frame}

            except Exception as e:
                logger.warning(f"{type(self).__name__} inference failed: {e!r}. Falling back to simulated detection.")
                return self._fallback.scan_frame(image_buffer)

    class YOLOVisionEngine(_UltralyticsScanMixin, BaseVisionEngine):
        def __init__(self, weights_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
            # Resolution order: explicit constructor arg > YOLO_MODEL_PATH env var >
            # hardcoded local default. Mirrors OllamaMLEngine's OLLAMA_HOST pattern —
            # swapping models per environment (local vs. a bigger cloud/GPU variant)
            # is a config change, not a code change.
            # `os.getenv(key, default)` only applies `default` when the key is
            # *absent* — a key present-but-empty (which is exactly what Docker
            # Compose passes through when YOLO_MODEL_PATH is unset in .env) would
            # otherwise silently win over DEFAULT_LOCAL_WEIGHTS. Chaining `or`
            # instead treats "" and "unset" identically.
            self.weights_path = weights_path or os.getenv("YOLO_MODEL_PATH") or DEFAULT_LOCAL_WEIGHTS
            self.confidence_threshold = confidence_threshold
            # Runtime degrade path: reuse the mock engine if real weights fail to
            # load, instead of leaving the app with a crashed/None vision tracker.
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
            # Same "" vs "unset" fix as YOLOVisionEngine — critical here, since
            # without it a compose-injected empty/YOLO-flavored YOLO_MODEL_PATH
            # would make YOLOEVisionEngine try to load a closed-set checkpoint
            # through the open-vocabulary YOLOE class and fail outright.
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

    # Which engine backs VisionTracker — config, not code, same pattern as
    # YOLO_MODEL_PATH. VISION_MODEL_TYPE=yoloe switches to the open-vocabulary
    # engine without touching this file.
    _ENGINE_TYPES = {"yolo": YOLOVisionEngine, "yoloe": YOLOEVisionEngine}
    _selected_type = os.getenv("VISION_MODEL_TYPE", DEFAULT_VISION_MODEL_TYPE).strip().lower()
    VisionTracker = _ENGINE_TYPES.get(_selected_type, YOLOVisionEngine)

except ImportError as e:
    logger.critical(f"Vision dependencies unavailable: {e}")
    # Seamless substitution: App falls back cleanly to the mock engine
    VisionTracker = FallbackVisionEngine