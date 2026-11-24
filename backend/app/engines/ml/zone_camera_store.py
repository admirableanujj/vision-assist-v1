# app/engines/ml/zone_camera_store.py
"""
Qdrant-backed store for objects seen on the live zone camera stream.

Collection: "zone_cameras_embeddings" (COSINE, 512-d clip-ViT-B-32 — the same
space as "item_embeddings", so a crop seen on camera can be registered as an
item by reusing its vector instead of re-embedding it).

One point = one sighting of one object (a YOLO box above the 50% threshold)
on one camera in one zone. Payload:

    zone_id           int    — zones.id
    camera_id         str    — cameras.id (or a synthetic webcam id)
    label             str    — YOLO class name
    description       str    — human-readable sighting summary
    created_at        str    — ISO-8601 UTC
    created_ts        float  — epoch seconds (range-indexed, used for ordering)
    confidence_score  float  — YOLO confidence, 0..1
    box               list   — [x1, y1, x2, y2] in the 640x480 frame
    engine            str    — "yolo" | "finetuned"
    owner_id          int    — user who owns the zone (from the WS JWT)
    thumbnail_b64     str    — small base64 JPEG of the crop

The stream sends ~10 frames/s, so callers are expected to throttle (see
vision_routes.py) — this module does not.
"""
import io
import os
import time
import uuid
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_ZONE_COLLECTION = "zone_cameras_embeddings"
DEFAULT_DIM = 512
RRF_K = 60  # standard RRF constant (Cormack et al. 2009)
# Confidence is the primary signal; recency and frequency refine it.
RRF_WEIGHTS = {"confidence": 1.0, "recency": 0.5, "frequency": 0.5}


class QdrantZoneCameraStore:
    def __init__(self, embedder: Any, client: Any, collection: Optional[str] = None,
                 dim: int = DEFAULT_DIM, thumbnail_size: int = 160):
        self.embedder = embedder
        self.client = client
        self.collection = collection or os.getenv("QDRANT_ZONE_COLLECTION", DEFAULT_ZONE_COLLECTION)
        self.dim = dim
        self.thumbnail_size = thumbnail_size
        self.ensure_collection()

    # ------------------------------------------------------------------ setup
    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        try:
            exists = self.client.collection_exists(self.collection)
        except AttributeError:
            exists = self.collection in {c.name for c in self.client.get_collections().collections}
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
        for field, schema in (
            ("zone_id", PayloadSchemaType.INTEGER),
            ("camera_id", PayloadSchemaType.KEYWORD),
            ("label", PayloadSchemaType.KEYWORD),
            ("created_ts", PayloadSchemaType.FLOAT),  # range index -> enables order_by
        ):
            try:
                self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
            except Exception:
                pass  # already exists, or in-memory client

    # ------------------------------------------------------------------ write
    def add_detection(self, zone_id: int, camera_id: str, label: str, confidence: float,
                      crop: Any, box: Optional[List[float]] = None, engine: str = "",
                      owner_id: Optional[int] = None) -> str:
        """Embed one object crop and persist it as a sighting. Returns the point id."""
        from qdrant_client.models import PointStruct

        img = self.embedder._to_image(crop)
        vector = list(self.embedder.embed_image(img))
        now = datetime.now(timezone.utc)
        payload = {
            "zone_id": int(zone_id),
            "camera_id": str(camera_id),
            "label": label,
            "description": f"{label} seen by camera {camera_id} in zone {zone_id} "
                           f"({confidence:.0%} confidence, {engine or 'yolo'} engine)",
            "created_at": now.isoformat(),
            "created_ts": now.timestamp(),
            "confidence_score": round(float(confidence), 4),
            "box": [float(c) for c in box] if box else None,
            "engine": engine,
            "owner_id": owner_id,
        }
        thumb = self._thumbnail_b64(img)
        if thumb:
            payload["thumbnail_b64"] = thumb

        point_id = str(uuid.uuid4())
        self.client.upsert(collection_name=self.collection,
                           points=[PointStruct(id=point_id, vector=vector, payload=payload)])
        return point_id

    # ------------------------------------------------------------------- read
    def recent_detections(self, zone_id: int, limit: int = 200,
                          since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Most recent sightings in a zone, newest first (payload only, no vectors)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, OrderBy, Direction

        must = [FieldCondition(key="zone_id", match=MatchValue(value=int(zone_id)))]
        if since_ts:
            must.append(FieldCondition(key="created_ts", range=Range(gte=since_ts)))
        flt = Filter(must=must)
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection, scroll_filter=flt, limit=limit,
                with_payload=True, with_vectors=False,
                order_by=OrderBy(key="created_ts", direction=Direction.DESC),
            )
        except Exception:
            # order_by needs a range index (not available on every client/mode):
            # pull a bounded page and sort locally instead.
            points, _ = self.client.scroll(
                collection_name=self.collection, scroll_filter=flt, limit=max(limit * 5, 1000),
                with_payload=True, with_vectors=False,
            )
            points = sorted(points, key=lambda p: (p.payload or {}).get("created_ts", 0), reverse=True)[:limit]
        return [{"id": str(p.id), **(p.payload or {})} for p in points]

    def get_detection(self, point_id: str, with_vector: bool = True) -> Optional[Dict[str, Any]]:
        recs = self.client.retrieve(collection_name=self.collection, ids=[point_id],
                                    with_payload=True, with_vectors=with_vector)
        if not recs:
            return None
        r = recs[0]
        out = {"id": str(r.id), **(r.payload or {})}
        if with_vector:
            out["vector"] = list(r.vector) if r.vector is not None else None
        return out

    # ------------------------------------------------------------------ utils
    def _thumbnail_b64(self, img: Any) -> Optional[str]:
        try:
            thumb = img.copy()
            thumb.thumbnail((self.thumbnail_size, self.thumbnail_size))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"[WARN] zone thumbnail generation failed: {e}")
            return None


def rrf_top_objects(detections: List[Dict[str, Any]], top_k: int = 5, k: int = RRF_K) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion over the objects (labels) seen in a zone.

    Each label is ranked independently by three signals, then fused with
    score = sum(weight / (k + rank)), weights in RRF_WEIGHTS:
      1. confidence — best YOLO confidence_score for that label
      2. recency    — most recent sighting (created_ts)
      3. frequency  — number of sightings in the window
    Tied values share a rank; confidence carries the most weight and is also
    the final tie-breaker. The representative point returned for
    each label is its highest-confidence sighting (best thumbnail to register).
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for d in detections:
        label = d.get("label")
        if not label:
            continue
        g = groups.setdefault(label, {"best": d, "max_conf": -1.0, "latest": 0.0, "count": 0})
        conf = float(d.get("confidence_score") or 0.0)
        if conf > g["max_conf"]:
            g["max_conf"], g["best"] = conf, d
        g["latest"] = max(g["latest"], float(d.get("created_ts") or 0.0))
        g["count"] += 1

    if not groups:
        return []

    labels = list(groups)
    signals = {
        "confidence": {l: groups[l]["max_conf"] for l in labels},
        # 1 s resolution: labels persisted in the same batch are "equally recent".
        "recency": {l: int(groups[l]["latest"]) for l in labels},
        "frequency": {l: groups[l]["count"] for l in labels},
    }
    fused = {l: 0.0 for l in labels}
    for name, values in signals.items():
        # Competition ranking: tied values share a rank (1, 2, 2, 4, ...), so a
        # tie never hands one label an arbitrary advantage.
        distinct = sorted(set(values.values()), reverse=True)
        first_rank, seen = {}, 0
        for v in distinct:
            first_rank[v] = seen + 1
            seen += sum(1 for x in values.values() if x == v)
        for label, v in values.items():
            fused[label] += RRF_WEIGHTS[name] / (k + first_rank[v])

    ordered = sorted(labels, key=lambda l: (fused[l], groups[l]["max_conf"]), reverse=True)[:top_k]
    return [
        {
            **groups[l]["best"],
            "rrf_score": round(fused[l], 6),
            "sightings": groups[l]["count"],
            "last_seen_ts": groups[l]["latest"],
        }
        for l in ordered
    ]
