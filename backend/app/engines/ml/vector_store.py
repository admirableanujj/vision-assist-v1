# app/ml_engine/vector_store.py
"""
Qdrant-backed item vector store (SKETCH)

Persists the CLIP embeddings of registered items — from their text label and/or
their image — into Qdrant, so a lost-item query (text *or* photo) can be matched
against everything the user has registered. Because clip-ViT-B-32 encodes text
and images into the same 512-d space (see embedding_engine.py), a text query can
retrieve image-registered items and vice versa.

Design
------
- One Qdrant collection (default "item_embeddings", COSINE, 512-d).
- Each registered item can contribute up to two points:
    * a TEXT point  — embedding of "<item_name>. <description>"
    * an IMAGE point — embedding of the item's photo (or detected crop)
  Point IDs are deterministic uuid5(item_id + modality), so re-registering the
  same item UPSERTS (updates) its points instead of creating duplicates.
- Payload carries the metadata needed to render a result without a second DB
  hop: item_id, owner_id, item_name (the label), description, modality,
  created_at, and — optionally — a small base64 JPEG thumbnail so the image is
  retrievable straight from Qdrant.
- Every search is scoped to the requesting owner_id via a Qdrant filter, so one
  user never matches another user's belongings.

Image storage tradeoff (read before shipping)
----------------------------------------------
Storing the *pixels* in Qdrant (as a base64 thumbnail in the payload) keeps the
system single-store and is fine for small crops/thumbnails. For full-resolution
images, prefer an object store (S3/MinIO) or Postgres BYTEA and keep only a
reference (URL/key) in the payload — set store_thumbnails=False and pass
extra_payload={"image_url": ...}. The vector + label always live in Qdrant
regardless.

Config (matches docker-compose.yml)
-----------------------------------
- QDRANT_URL           (default http://qdrant:6333)  — or QDRANT_HOST/QDRANT_PORT
- QDRANT_API_KEY       (or QDRANT__SERVICE__API_KEY, or QDRANT_API_KEY_FILE for a
                        Docker-secret file, matching POSTGRES_PASSWORD_FILE)
- QDRANT_COLLECTION    (default "item_embeddings")

Usage
-----
    from ml_engine import QdrantItemStore
    store = QdrantItemStore()          # reads CLIP engine + Qdrant config from env
    store.index_item(
        item_id="item_7_wallet",
        owner_id=7,
        item_name="wallet",
        description="black leather bifold",
        image=cam_frame,               # optional: PIL image / ndarray / bytes / path
    )
    hits = store.search_by_text("black wallet", owner_id=7, limit=5)

__original_author__ = "Venkata Balaji Yadalla"
__license__ = "MIT"
"""
__author__ = "Venkata Balaji Yadalla"
__license__ = "MIT"
__version__ = "0.1.0"  # sketch

import io
import os
import uuid
import base64
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Any, Dict

# Stable namespace so uuid5 point IDs are reproducible across processes/runs.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

DEFAULT_COLLECTION = "item_embeddings"
DEFAULT_DIM = 512  # clip-ViT-B-32


def _point_id(item_id: str, modality: str) -> str:
    """Deterministic point ID: same (item_id, modality) always maps to one point."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{item_id}:{modality}"))


def _read_secret_file(path: Optional[str]) -> Optional[str]:
    """Read a Docker-secret-style file (e.g. /run/secrets/qdrant_api_key)."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


class QdrantItemStore:
    """
    Thin service that turns registered items into Qdrant points and answers
    similarity queries. The embedder and Qdrant client are injectable so this is
    unit-testable without a live server or the CLIP weights.
    """

    def __init__(
        self,
        embedder: Any = None,
        client: Any = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        dim: int = DEFAULT_DIM,
        store_thumbnails: bool = True,
        thumbnail_size: int = 224,
    ):
        # --- Embedder: reuse the shared CLIP engine unless one is injected. ---
        if embedder is None:
            # Local import keeps this module importable even where embedding deps
            # are stubbed (tests), and avoids a hard import cycle.
            from .embedding_engine import CLIPEmbeddingEngine
            embedder = CLIPEmbeddingEngine()
        self.embedder = embedder

        self.collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
        self.dim = dim
        self.store_thumbnails = store_thumbnails
        self.thumbnail_size = thumbnail_size

        # --- Qdrant client: injected (e.g. in-memory for tests) or from env. ---
        if client is None:
            from qdrant_client import QdrantClient
            host = os.getenv("QDRANT_HOST", "qdrant")
            port = os.getenv("QDRANT_PORT", "6333")
            url = url or os.getenv("QDRANT_URL", f"http://{host}:{port}")
            # Key resolution mirrors the repo's secret-file convention
            # (cf. POSTGRES_PASSWORD_FILE): explicit arg > env > secret file.
            api_key = (
                api_key
                or os.getenv("QDRANT_API_KEY")
                or os.getenv("QDRANT__SERVICE__API_KEY")
                or _read_secret_file(os.getenv("QDRANT_API_KEY_FILE"))
            )
            client = QdrantClient(url=url, api_key=api_key, timeout=10.0)
        self.client = client

        self.ensure_collection()

    # ------------------------------------------------------------------ setup
    def ensure_collection(self) -> None:
        """Create the collection once if it doesn't already exist."""
        from qdrant_client.models import Distance, VectorParams

        exists = False
        try:
            exists = self.client.collection_exists(self.collection)
        except AttributeError:
            # Older clients: fall back to listing.
            names = {c.name for c in self.client.get_collections().collections}
            exists = self.collection in names

        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------ write
    def index_item(
        self,
        item_id: str,
        owner_id: int,
        item_name: str,
        description: str = "",
        image: Any = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Upsert the text and (optional) image points for one item.
        Returns the list of point IDs written.
        """
        from qdrant_client.models import PointStruct

        created_at = datetime.now(timezone.utc).isoformat()
        base_payload = {
            "item_id": item_id,
            "owner_id": owner_id,
            "item_name": item_name,
            "description": description,
            "created_at": created_at,
        }
        if extra_payload:
            base_payload.update(extra_payload)

        points: List[PointStruct] = []

        # TEXT point — label + description in one string.
        text_blob = item_name if not description else f"{item_name}. {description}"
        if text_blob.strip():
            tvec = self.embedder.embed_text(text_blob)
            points.append(PointStruct(
                id=_point_id(item_id, "text"),
                vector=list(tvec),
                payload={**base_payload, "modality": "text", "text": text_blob},
            ))

        # IMAGE point — CLIP image embedding, plus an optional thumbnail so the
        # picture itself is recoverable from Qdrant.
        if image is not None:
            ivec = self.embedder.embed_image(image)
            img_payload = {**base_payload, "modality": "image"}
            if self.store_thumbnails:
                thumb = self._thumbnail_b64(image)
                if thumb:
                    img_payload["thumbnail_b64"] = thumb
            points.append(PointStruct(
                id=_point_id(item_id, "image"),
                vector=list(ivec),
                payload=img_payload,
            ))

        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return [p.id for p in points]

    # ----------------------------------------------------------------- search
    def search_by_text(self, text: str, owner_id: Optional[int] = None, limit: int = 5) -> List[Dict[str, Any]]:
        return self._search(self.embedder.embed_text(text), owner_id, limit)

    def search_by_image(self, image: Any, owner_id: Optional[int] = None, limit: int = 5) -> List[Dict[str, Any]]:
        return self._search(self.embedder.embed_image(image), owner_id, limit)

    def _search(self, vector: Sequence[float], owner_id: Optional[int], limit: int) -> List[Dict[str, Any]]:
        flt = self._owner_filter(owner_id)
        # query_points is the modern API (qdrant-client >= 1.10); fall back to
        # the deprecated .search() so this stays correct on older client pins.
        try:
            resp = self.client.query_points(
                collection_name=self.collection, query=list(vector),
                query_filter=flt, limit=limit, with_payload=True,
            )
            hits = resp.points
        except AttributeError:
            hits = self.client.search(
                collection_name=self.collection, query_vector=list(vector),
                query_filter=flt, limit=limit, with_payload=True,
            )
        return [{"id": h.id, "score": h.score, **(h.payload or {})} for h in hits]

    @staticmethod
    def _owner_filter(owner_id: Optional[int]):
        if owner_id is None:
            return None
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        return Filter(must=[FieldCondition(key="owner_id", match=MatchValue(value=owner_id))])

    # ----------------------------------------------------------------- delete
    def delete_item(self, item_id: str) -> None:
        """Remove both the text and image points for an item."""
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[
                _point_id(item_id, "text"), _point_id(item_id, "image"),
            ]),
        )

    # ------------------------------------------------------------------ utils
    def _thumbnail_b64(self, image: Any) -> Optional[str]:
        """Downscale an image to a small JPEG and base64-encode it for the payload."""
        try:
            from PIL import Image
            img = self.embedder._to_image(image)  # reuse the engine's normalizer
            img.thumbnail((self.thumbnail_size, self.thumbnail_size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:  # never let thumbnailing break indexing
            print(f"[WARN] thumbnail generation failed: {e}")
            return None
