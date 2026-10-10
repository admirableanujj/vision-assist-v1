# app/ml_engine/embedding_engine.py
"""
CLIP Multimodal Embedding Engine

Wraps the `clip-ViT-B-32` SentenceTransformer to turn both text (item names,
descriptions, spoken queries) and images (camera frames / detected object crops)
into a shared 512-dimensional vector space. Because CLIP encodes text and images
into the *same* space, a text query like "black leather wallet" can be compared
directly against the embedding of a detected object crop — the core primitive
behind semantic Lost & Found matching and the `item_embeddings` table / Qdrant
vector store.

Follows the same ABC → concrete pattern as the rest of the codebase
(`BaseMLEngine`, `BaseVisionEngine`) and the same "config-not-code" model
selection convention as `OllamaMLEngine` (OLLAMA_HOST) and `YOLOVisionEngine`
(YOLO_MODEL_PATH): the model is resolved from an explicit constructor arg, then
the `EMBEDDING_MODEL_NAME` env var, then the hardcoded default. Swapping to a
larger CLIP variant on a GPU deployment is therefore a config change, not a
code change.

Usage:
    from ml_engine import CLIPEmbeddingEngine
    embedder = CLIPEmbeddingEngine()
    vec = embedder.embed_text("black leather wallet")   # -> list[float], len 512
    img_vec = embedder.embed_image("crop.jpg")          # -> list[float], len 512
    score = embedder.similarity(vec, img_vec)           # cosine, in [-1, 1]

Dependencies:
    sentence-transformers
    torch
    pillow (pulled in by sentence-transformers)

__original_author__ = "Venkata Balaji Yadalla"
__license__ = "MIT"
"""
__author__ = "Venkata Balaji Yadalla"
__license__ = "MIT"
__version__ = "1.0.0"

import os
from abc import ABC, abstractmethod
from typing import List, Sequence, Union

# clip-ViT-B-32 emits a 512-d vector for both text and images.
DEFAULT_EMBEDDING_MODEL = "clip-ViT-B-32"
DEFAULT_EMBEDDING_DIM = 512

# Type alias for the many shapes an image can arrive in.
ImageLike = Union[str, bytes, bytearray, "object"]


class BaseEmbeddingEngine(ABC):
    """
    Abstract contract for embedding services. Any backend (CLIP today, a
    different multimodal encoder tomorrow) must expose text + image encoders
    that return plain Python float lists so callers stay decoupled from the
    underlying tensor framework.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Encode a single string into an embedding vector."""
        pass  # pragma: no cover

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch of strings into embedding vectors."""
        pass  # pragma: no cover

    @abstractmethod
    def embed_image(self, image: ImageLike) -> List[float]:
        """Encode a single image into an embedding vector."""
        pass  # pragma: no cover

    @abstractmethod
    def embed_images(self, images: Sequence[ImageLike]) -> List[List[float]]:
        """Encode a batch of images into embedding vectors."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors produced by this engine."""
        pass  # pragma: no cover

    @staticmethod
    def similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """
        Cosine similarity between two vectors. With normalized embeddings this
        is just the dot product, but we normalize defensively so the helper is
        correct regardless of how the vectors were produced.
        """
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# --- REAL CLIP SYSTEM BOUNDARY ---
try:
    from sentence_transformers import SentenceTransformer

    class CLIPEmbeddingEngine(BaseEmbeddingEngine):
        """
        Concrete embedding engine backed by a `clip-ViT-B-32` SentenceTransformer.

        The model is loaded lazily on first use so importing this module (and
        constructing the engine inside Streamlit's cached boot path) never pays
        the model-load cost until an embedding is actually requested.
        """

        def __init__(self, model_name: str = None, normalize: bool = True):
            # Resolution order mirrors OllamaMLEngine / YOLOVisionEngine:
            # explicit arg > EMBEDDING_MODEL_NAME env var > hardcoded default.
            self.model_name = model_name or os.getenv(
                "EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL
            )
            # Normalized vectors make cosine similarity a plain dot product and
            # are what Qdrant's Distance.COSINE expects for stable scoring.
            self.normalize = normalize
            self._model = None  # lazy
            self._dim = DEFAULT_EMBEDDING_DIM
            print(f"[INFO] CLIPEmbeddingEngine configured for model '{self.model_name}' (lazy load).")

        @property
        def model(self) -> "SentenceTransformer":
            """Lazily load and cache the underlying SentenceTransformer."""
            if self._model is None:
                print(f"[INFO] Loading embedding model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                try:
                    self._dim = int(self._model.get_sentence_embedding_dimension())
                except Exception:
                    self._dim = DEFAULT_EMBEDDING_DIM
                print(f"[INFO] Embedding model ready (dim={self._dim}).")
            return self._model

        @property
        def dimension(self) -> int:
            # Touch the model so we report the true dimension once loaded; before
            # first load we return the known clip-ViT-B-32 default.
            if self._model is None:
                return self._dim
            return self._dim

        def _encode(self, payload, batch: bool):
            """Shared encode path returning python lists (not tensors/ndarrays)."""
            vectors = self.model.encode(
                payload,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if batch:
                return [v.tolist() for v in vectors]
            return vectors.tolist()

        def embed_text(self, text: str) -> List[float]:
            if not text:
                return [0.0] * self.dimension
            return self._encode(text, batch=False)

        def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
            if not texts:
                return []
            return self._encode(list(texts), batch=True)

        def _to_image(self, image: ImageLike):
            """
            Normalize the many shapes an image can arrive in into a PIL.Image
            that SentenceTransformer's CLIP wrapper accepts:
              - str/os.PathLike -> opened from disk
              - raw bytes / file-like (e.g. Streamlit UploadedFile) -> decoded
              - numpy.ndarray -> treated as RGB (note: OpenCV frames are BGR,
                convert with cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) first)
              - PIL.Image -> passed through
            """
            from PIL import Image  # local import: only needed on the image path

            # Already a PIL image
            if isinstance(image, Image.Image):
                return image.convert("RGB")

            # Filesystem path
            if isinstance(image, (str, os.PathLike)):
                return Image.open(image).convert("RGB")

            # numpy array (assumed RGB)
            try:
                import numpy as np
                if isinstance(image, np.ndarray):
                    return Image.fromarray(image).convert("RGB")
            except ImportError:
                pass

            # Raw bytes
            if isinstance(image, (bytes, bytearray)):
                import io
                return Image.open(io.BytesIO(bytes(image))).convert("RGB")

            # File-like object (has .getvalue() e.g. Streamlit, or .read())
            if hasattr(image, "getvalue"):
                import io
                return Image.open(io.BytesIO(image.getvalue())).convert("RGB")
            if hasattr(image, "read"):
                import io
                return Image.open(io.BytesIO(image.read())).convert("RGB")

            raise TypeError(f"Unsupported image type for embedding: {type(image)!r}")

        def embed_image(self, image: ImageLike) -> List[float]:
            if image is None:
                return [0.0] * self.dimension
            return self._encode(self._to_image(image), batch=False)

        def embed_images(self, images: Sequence[ImageLike]) -> List[List[float]]:
            if not images:
                return []
            pil_images = [self._to_image(img) for img in images]
            return self._encode(pil_images, batch=True)

    EmbeddingEncoder = CLIPEmbeddingEngine

except ImportError as e:  # pragma: no cover
    # Seamless degrade: if sentence-transformers isn't installed (e.g. the local
    # test environment that stubs Docker-only deps), expose a stub that
    # constructs fine but refuses to fabricate vectors — silently faking
    # embeddings would corrupt any vector store they're written to.
    print(f"[CRITICAL] Embedding dependencies unavailable: {e}. "
          f"CLIPEmbeddingEngine will raise on use until sentence-transformers is installed.")

    class CLIPEmbeddingEngine(BaseEmbeddingEngine):  # type: ignore[no-redef]
        def __init__(self, model_name: str = None, normalize: bool = True):
            self.model_name = model_name or os.getenv(
                "EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL
            )
            self.normalize = normalize
            self._dim = DEFAULT_EMBEDDING_DIM

        def _unavailable(self):
            raise RuntimeError(
                "sentence-transformers is not installed, so the CLIP embedding "
                "model is unavailable. Install it (see requirements.txt) or "
                "rebuild the Docker image to enable embeddings."
            )

        @property
        def dimension(self) -> int:
            return self._dim

        def embed_text(self, text: str) -> List[float]:
            self._unavailable()

        def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
            self._unavailable()

        def embed_image(self, image: ImageLike) -> List[float]:
            self._unavailable()

        def embed_images(self, images: Sequence[ImageLike]) -> List[List[float]]:
            self._unavailable()

    EmbeddingEncoder = CLIPEmbeddingEngine
