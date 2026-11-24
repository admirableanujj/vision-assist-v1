# app/engines/ml/qdrant_service.py
"""
Process-wide singletons for the CLIP embedder, the Qdrant client and the two
collections the app uses:

    item_embeddings          -> QdrantItemStore        (registered items)
    zone_cameras_embeddings  -> QdrantZoneCameraStore  (live stream sightings)

Everything is built lazily on first use so the API still boots (and the
camera stream still works) if Qdrant or the CLIP weights are unavailable —
callers catch the exception and degrade.
"""
import os
import threading
from typing import Any, Optional

from .vector_store import QdrantItemStore, _read_secret_file
from .zone_camera_store import QdrantZoneCameraStore

_lock = threading.Lock()
_embedder: Optional[Any] = None
_client: Optional[Any] = None
_item_store: Optional[QdrantItemStore] = None
_zone_store: Optional[QdrantZoneCameraStore] = None


def get_embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            from .embedding_engine import CLIPEmbeddingEngine
            _embedder = CLIPEmbeddingEngine()
        return _embedder


def get_qdrant_client():
    global _client
    with _lock:
        if _client is None:
            from qdrant_client import QdrantClient
            host = os.getenv("QDRANT_HOST", "qdrant")
            port = os.getenv("QDRANT_PORT", "6333")
            url = os.getenv("QDRANT_URL", f"http://{host}:{port}")
            api_key = (
                os.getenv("QDRANT_API_KEY")
                or os.getenv("QDRANT__SERVICE__API_KEY")
                or _read_secret_file(os.getenv("QDRANT_API_KEY_FILE", "/run/secrets/qdrant_api_key"))
            )
            _client = QdrantClient(url=url, api_key=api_key, timeout=10.0)
        return _client


def get_item_store() -> QdrantItemStore:
    global _item_store
    if _item_store is None:
        embedder, client = get_embedder(), get_qdrant_client()
        with _lock:
            if _item_store is None:
                store = QdrantItemStore(embedder=embedder, client=client)
                store.ensure_payload_indexes()
                _item_store = store
    return _item_store


def get_zone_camera_store() -> QdrantZoneCameraStore:
    global _zone_store
    if _zone_store is None:
        embedder, client = get_embedder(), get_qdrant_client()
        with _lock:
            if _zone_store is None:
                _zone_store = QdrantZoneCameraStore(embedder=embedder, client=client)
    return _zone_store


def thumbnail_data_url(payload: dict) -> Optional[str]:
    """
    Qdrant stores images as base64 JPEG thumbnails; React's <img> needs a data
    URL. If a payload ever carries an object-store URL (S3/GCS, see the
    image-storage note in vector_store.py) that wins.
    """
    if not payload:
        return None
    if payload.get("image_url"):
        return payload["image_url"]
    b64 = payload.get("thumbnail_b64")
    return f"data:image/jpeg;base64,{b64}" if b64 else None
