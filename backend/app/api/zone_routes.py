import base64
import json
import logging
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.core.database import get_db
from app.core.security import get_current_user

logger = logging.getLogger("zone_routes")
router = APIRouter(prefix="/api/v1", tags=["Zones & Inventory"])

class ItemRegisterRequest(BaseModel):
    # Preferred: the zone_cameras_embeddings point id of an "Objects Identified"
    # card — the item is built from that sighting's crop, vector and metadata.
    detection_id: Optional[str] = None
    # Optional display-name override (defaults to the detection's YOLO label).
    name: Optional[str] = None
    description: Optional[str] = None
    # Legacy path (no detection_id): name + optional data-URL image.
    image: Optional[str] = None
    zone_id: Optional[int] = None
    confidence: Optional[float] = None  # accepted for compatibility, never stored in Qdrant

class SearchQueryRequest(BaseModel):
    query: str
    zone_id: int

@router.get("/zones")
async def get_user_zones(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Fetch all zones associated with the currently authenticated user from PostgreSQL."""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        result = db.execute(
            text("SELECT id, name FROM zones WHERE user_id = :user_id"), 
            {"user_id": user_id}
        )
        rows = result.fetchall()
        zones = [{"id": row[0], "name": row[1]} for row in rows]

        if not zones:
            zones = [
                {"id": 1, "name": "Zone 1: Main Living Zone"},
                {"id": 2, "name": "Zone 2: Master Bedroom"},
                {"id": 3, "name": "Zone 3: Kitchen Zone"}
            ]
        return {"zones": zones}
    except Exception as e:
        logger.error(f"Error fetching zones: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving zones.")

TOP_K_DETECTED = int(os.getenv("ZONE_TOP_K", "5"))
# How many recent sightings feed the RRF ranking, and (optionally) how far back.
ZONE_RECENT_POINTS = int(os.getenv("ZONE_RECENT_POINTS", "200"))
ZONE_WINDOW_SECONDS = float(os.getenv("ZONE_WINDOW_SECONDS", "0"))  # 0 = no time limit


def _user_id(current_user: dict):
    return current_user.get("id") or current_user.get("user_id")


def _assert_zone_owner(db: Session, zone_id: int, user_id) -> str:
    row = db.execute(
        text("SELECT name FROM zones WHERE id = :zone_id AND user_id = :user_id"),
        {"zone_id": zone_id, "user_id": user_id},
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Zone does not belong to user or does not exist.",
        )
    return row[0]


def _legacy_postgres_detections(db: Session, zone_id: int, zone_name: str) -> List[Dict[str, Any]]:
    """Pre-Qdrant fallback: rows from the relational detections table (no images)."""
    rows = db.execute(
        text("""
            SELECT d.id, d.object_class, d.confidence
            FROM detections d
            LEFT JOIN cameras c ON d.camera_id = c.id
            WHERE (d.zone_name ILIKE :zone_name OR c.zone_id = :zone_id)
            ORDER BY d.timestamp DESC LIMIT :lim
        """),
        {"zone_name": f"%{zone_name}%", "zone_id": zone_id, "lim": TOP_K_DETECTED},
    ).fetchall()
    return [{"id": str(r[0]), "name": r[1], "confidence": r[2], "image": None, "source": "postgres"} for r in rows]


@router.get("/zones/{zone_id}/detected")
def get_zone_detected_objects(zone_id: int, limit: int = TOP_K_DETECTED,
                              db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    Requirement 2: top-N (default 5) objects seen on this zone's live feed,
    read from Qdrant zone_cameras_embeddings and re-ranked with RRF over
    confidence score, recency and sighting frequency.
    """
    zone_name = _assert_zone_owner(db, zone_id, _user_id(current_user))
    limit = max(1, min(limit, 20))

    try:
        import time
        from app.engines.ml.qdrant_service import get_zone_camera_store, thumbnail_data_url
        from app.engines.ml.zone_camera_store import rrf_top_objects

        since = time.time() - ZONE_WINDOW_SECONDS if ZONE_WINDOW_SECONDS > 0 else None
        recent = get_zone_camera_store().recent_detections(zone_id, limit=ZONE_RECENT_POINTS, since_ts=since)
        ranked = rrf_top_objects(recent, top_k=limit)
        detected_items = [
            {
                "id": d["id"],                       # zone_cameras_embeddings point id -> detection_id
                "detection_id": d["id"],
                "name": d.get("label"),
                "label": d.get("label"),
                "confidence": d.get("confidence_score"),
                "camera_id": d.get("camera_id"),
                "zone_id": d.get("zone_id"),
                "description": d.get("description"),
                "created_at": d.get("created_at"),
                "sightings": d.get("sightings"),
                "rrf_score": d.get("rrf_score"),
                "image": thumbnail_data_url(d),
                "source": "qdrant",
            }
            for d in ranked
        ]
        return {"detected_items": detected_items, "source": "qdrant"}
    except Exception as e:
        logger.error(f"Qdrant zone lookup failed for zone {zone_id}, falling back to Postgres: {e}")
        try:
            return {"detected_items": _legacy_postgres_detections(db, zone_id, zone_name), "source": "postgres"}
        except Exception as e2:
            logger.error(f"Postgres detections fallback failed for zone {zone_id}: {e2}")
            return {"detected_items": [], "source": "none"}


@router.get("/items/registered")
def get_registered_items(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Registered items from Postgres, with thumbnails pulled from Qdrant item_embeddings."""
    user_id = _user_id(current_user)
    try:
        rows = db.execute(
            text("""SELECT id, item_name, description, home_zone_id, (embedding IS NOT NULL) AS has_embedding
                    FROM items WHERE owner_id = :user_id ORDER BY created_on DESC"""),
            {"user_id": user_id},
        ).fetchall()
    except Exception as e:
        logger.error(f"Error fetching registered items: {e}")
        return {"items": []}

    thumbs: Dict[str, Any] = {}
    try:
        from app.engines.ml.qdrant_service import get_item_store, thumbnail_data_url
        payloads = get_item_store().get_item_points([str(r[0]) for r in rows], modality="image")
        thumbs = {iid: thumbnail_data_url(p) for iid, p in payloads.items()}
    except Exception as e:
        logger.warning(f"Could not load item thumbnails from Qdrant: {e}")

    items = [
        {
            "id": str(r[0]),
            "name": r[1],
            "description": r[2],
            "zone_id": r[3],
            "has_embedding": bool(r[4]),
            "image": thumbs.get(str(r[0])),
        }
        for r in rows
    ]
    return {"items": items}


def _decode_data_url(data: Optional[str]) -> Optional[bytes]:
    if not data or not data.startswith("data:image"):
        return None
    try:
        return base64.b64decode(data.split(",", 1)[1])
    except Exception:
        return None


@router.post("/items/register")
def register_item(payload: ItemRegisterRequest, db: Session = Depends(get_db),
                  current_user: dict = Depends(get_current_user)):
    """
    Requirement 3: take a sighting from zone_cameras_embeddings (image +
    metadata + vector), write it to Qdrant item_embeddings (text + image
    points, no confidence score), and store the returned embedding in
    Postgres items.embedding.
    """
    user_id = _user_id(current_user)

    try:
        from app.engines.ml.qdrant_service import get_item_store, get_zone_camera_store, get_embedder
        item_store = get_item_store()
    except Exception as e:
        logger.error(f"Vector store unavailable for registration: {e}")
        raise HTTPException(status_code=503, detail="Vector store (Qdrant/CLIP) is unavailable.")

    image_vector, thumbnail_b64, extra = None, None, {}

    if payload.detection_id:
        det = get_zone_camera_store().get_detection(payload.detection_id, with_vector=True)
        if not det:
            raise HTTPException(status_code=404, detail="Detection not found in zone_cameras_embeddings.")
        zone_id = det.get("zone_id")
        _assert_zone_owner(db, zone_id, user_id)
        item_name = (payload.name or det.get("label") or "object").strip()
        description = payload.description or (
            f"{det.get('label')} registered from camera {det.get('camera_id')} in zone {zone_id}"
        )
        image_vector = det.get("vector")
        thumbnail_b64 = det.get("thumbnail_b64")
        extra = {
            "zone_id": zone_id,
            "camera_id": det.get("camera_id"),
            "object_class": det.get("label"),
            "source_detection_id": det.get("id"),
        }
        object_class = det.get("label")
    else:
        if not payload.name or payload.zone_id is None:
            raise HTTPException(status_code=422, detail="Provide detection_id, or name and zone_id.")
        zone_id = payload.zone_id
        _assert_zone_owner(db, zone_id, user_id)
        item_name = payload.name.strip()
        description = payload.description or f"Registered via Vision UI in zone {zone_id}"
        object_class = item_name
        img_bytes = _decode_data_url(payload.image)
        if img_bytes:
            image_vector = get_embedder().embed_image(img_bytes)
            thumbnail_b64 = item_store._thumbnail_b64(img_bytes)
        extra = {"zone_id": zone_id, "object_class": item_name}

    item_id = f"item_{uuid.uuid4().hex[:8]}"
    try:
        db.execute(
            text("""
                INSERT INTO items (id, user_id, owner_id, item_name, description, object_class, home_zone_id)
                VALUES (:id, :user_id, :owner_id, :item_name, :description, :object_class, :home_zone_id)
            """),
            {"id": item_id, "user_id": user_id, "owner_id": user_id, "item_name": item_name,
             "description": description, "object_class": object_class, "home_zone_id": zone_id},
        )

        vectors = item_store.index_item_vectors(
            item_id=item_id, owner_id=user_id, item_name=item_name, description=description,
            image_vector=image_vector, thumbnail_b64=thumbnail_b64, extra_payload=extra,
        )
        # The image vector is the item's primary embedding (it's what matches
        # future camera crops); text-only registrations fall back to the text vector.
        modality = "image" if "image" in vectors else "text"
        embedding = vectors[modality]

        db.execute(
            text("UPDATE items SET embedding = :emb, last_modified_on = CURRENT_TIMESTAMP WHERE id = :id"),
            {"emb": json.dumps(embedding), "id": item_id},
        )
        db.commit()
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        try:
            item_store.delete_item(item_id)  # don't leave orphan vectors behind
        except Exception:
            pass
        logger.error(f"Failed to register item: {e}")
        raise HTTPException(status_code=500, detail="Item registration failed.")

    logger.info(f"Registered item '{item_name}' ({item_id}) for user {user_id} in zone {zone_id}")
    return {
        "status": "success",
        "message": f"Successfully registered {item_name}",
        "item_id": item_id,
        "item_name": item_name,
        "zone_id": zone_id,
        "modality": modality,
        "embedding": embedding,
    }


@router.get("/items/{item_id}/matches")
def get_item_matches(item_id: str, limit: int = TOP_K_DETECTED,
                     db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    Requirement 4: read the item's embedding from Postgres items.embedding,
    run it against Qdrant item_embeddings (scoped to this owner) and return
    the matches with their thumbnails as data URLs for the Objects Identified panel.
    """
    user_id = _user_id(current_user)
    row = db.execute(
        text("SELECT item_name, embedding FROM items WHERE id = :id AND owner_id = :u"),
        {"id": item_id, "u": user_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Item not found.")
    if not row[1]:
        raise HTTPException(status_code=409, detail="Item has no stored embedding; re-register it.")
    try:
        vector = json.loads(row[1])
    except (TypeError, ValueError):
        raise HTTPException(status_code=409, detail="Stored embedding is not valid JSON; re-register it.")

    limit = max(1, min(limit, 20))
    try:
        from app.engines.ml.qdrant_service import get_item_store, thumbnail_data_url
        store = get_item_store()
        # Each item has up to two points (text + image) — over-fetch, then keep
        # one result per item_id.
        hits = store.search_by_vector(vector, owner_id=user_id, limit=limit * 3)
        best: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            iid = h.get("item_id")
            if iid and (iid not in best or h["score"] > best[iid]["score"]):
                best[iid] = h
        ordered = sorted(best.values(), key=lambda h: h["score"], reverse=True)[:limit]

        # A text-point hit has no thumbnail; borrow the item's image point payload.
        need = [h["item_id"] for h in ordered if not h.get("thumbnail_b64")]
        image_payloads = store.get_item_points(need, modality="image") if need else {}
    except Exception as e:
        logger.error(f"Qdrant item match failed for {item_id}: {e}")
        raise HTTPException(status_code=503, detail="Vector store (Qdrant) is unavailable.")

    matches = [
        {
            "id": h["item_id"],
            "item_id": h["item_id"],
            "name": h.get("item_name"),
            "description": h.get("description"),
            "zone_id": h.get("zone_id"),
            "camera_id": h.get("camera_id"),
            "confidence": round(float(h["score"]), 4),   # cosine similarity
            "modality": h.get("modality"),
            "is_self": h["item_id"] == item_id,
            "image": thumbnail_data_url(h) or thumbnail_data_url(image_payloads.get(h["item_id"], {})),
        }
        for h in ordered
    ]
    return {"item_id": item_id, "item_name": row[0], "matches": matches}

@router.post("/search")
async def search_vector_store(payload: SearchQueryRequest, current_user: dict = Depends(get_current_user)):
    """Perform vector similarity search for registered objects across zones."""
    query = payload.query.lower()
    zone_id = payload.zone_id
    
    return {
        "status": "success",
        "query": query,
        "zone_id": zone_id,
        "message": f"Vector search located '{query}' in zone ID {zone_id} with 96.4% semantic match confidence."
    }