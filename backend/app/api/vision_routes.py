from fastapi import APIRouter, WebSocket, WebSocketDisconnect,Query
import asyncio
import io
import logging
import os
import time
from typing import Optional

import jwt
from sqlalchemy import text

from app.core.database import SessionLocal
from app.core.security import SECRET_KEY, ALGORITHM
from app.engines.vision import get_engine, resolve_engine_key

logger = logging.getLogger("vision_routes")
router = APIRouter(tags=["Vision & Object Detection"])

# App-facing choice is deliberately just these two, so a user can clearly
# compare base vs. fine-tuned detection on the same scene. YOLOE still works
# as a config-only option (VISION_MODEL_TYPE=yoloe) for anyone who wants it,
# just not exposed here or in the frontend toggle.
_VALID_ENGINES = {"yolo", "finetuned"}

# FallbackVisionEngine reports box=None (nothing real to draw around), but the
# frontend's canvas overlay destructures `box` unconditionally with no null
# check — substitute a fixed placeholder so mock detections stay visible
# instead of crashing the overlay.
_MOCK_BOX = [20, 20, 160, 160]

def _wire_safe_detections(detections: list) -> list:
    return [
        {**d, "box": d["box"] or _MOCK_BOX}
        for d in detections
    ]

# --------------------------------------------------------------------------
# Persisting stream detections to Qdrant (zone_cameras_embeddings)
# --------------------------------------------------------------------------
# The stream runs at ~10 fps, so embedding every box of every frame would both
# swamp Qdrant and stall the socket on CPU CLIP inference. Instead each
# (zone, camera, label) is persisted at most once per ZONE_EMBED_INTERVAL_S,
# keeping that window's highest-confidence box, and the CLIP work runs in a
# worker thread so detection replies are never delayed by it.
ZONE_EMBED_INTERVAL_S = float(os.getenv("ZONE_EMBED_INTERVAL_S", "5"))
ZONE_EMBED_ENABLED = os.getenv("ZONE_EMBED_ENABLED", "true").lower() not in ("0", "false", "no")
_CROP_PAD = 0.08  # pad each box by 8% so the crop keeps a little context


def _decode_user(token: Optional[str]) -> Optional[dict]:
    """Best-effort JWT decode for the WS (browsers can't set auth headers on WebSockets)."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("user_id") is None:
            return None
        return {"id": payload["user_id"], "username": payload.get("sub")}
    except jwt.PyJWTError:
        return None


def _resolve_zone_camera(user_id: int, zone: str, camera: Optional[str]):
    """
    Validate the zone belongs to the user and pick the camera id to record.
    Returns (zone_id, camera_id) or (None, None) if persistence isn't allowed.
    """
    try:
        zone_id = int(str(zone).strip())
    except (TypeError, ValueError):
        return None, None
    db = SessionLocal()
    try:
        owned = db.execute(
            text("SELECT 1 FROM zones WHERE id = :z AND user_id = :u"), {"z": zone_id, "u": user_id}
        ).fetchone()
        if not owned:
            return None, None
        if camera:
            row = db.execute(
                text("SELECT id FROM cameras WHERE id = :c AND zone_id = :z"), {"c": camera, "z": zone_id}
            ).fetchone()
            if row:
                return zone_id, row[0]
        row = db.execute(
            text("SELECT id FROM cameras WHERE zone_id = :z ORDER BY (owner_id = :u) DESC, id LIMIT 1"),
            {"z": zone_id, "u": user_id},
        ).fetchone()
        # The browser webcam isn't a registered camera — give it a stable synthetic id.
        return zone_id, (row[0] if row else f"webcam_zone_{zone_id}")
    except Exception as e:
        logger.warning(f"Zone/camera lookup failed, detections won't be persisted: {e}")
        return None, None
    finally:
        db.close()


def _persist_detections(frame: bytes, detections: list, zone_id: int, camera_id: str,
                        engine_key: str, owner_id: int) -> None:
    """Worker-thread body: crop each box, CLIP-embed it, upsert to zone_cameras_embeddings."""
    from PIL import Image
    from app.engines.ml.qdrant_service import get_zone_camera_store

    store = get_zone_camera_store()
    img = Image.open(io.BytesIO(frame)).convert("RGB")
    W, H = img.size
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        pw, ph = (x2 - x1) * _CROP_PAD, (y2 - y1) * _CROP_PAD
        crop_box = (max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(W, int(x2 + pw)), min(H, int(y2 + ph)))
        if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < 4:
            continue
        store.add_detection(
            zone_id=zone_id, camera_id=camera_id, label=d["label"],
            confidence=d["confidence"], crop=img.crop(crop_box), box=list(d["box"]),
            engine=engine_key, owner_id=owner_id,
        )


class _DetectionPersister:
    """Per-connection throttle + single in-flight background job."""

    def __init__(self, zone_id: int, camera_id: str, engine_key: str, owner_id: int):
        self.zone_id, self.camera_id = zone_id, camera_id
        self.engine_key, self.owner_id = engine_key, owner_id
        self._last_saved: dict = {}  # label -> monotonic ts
        self._task: Optional[asyncio.Future] = None
        self._disabled = False

    def offer(self, frame: bytes, detections: list) -> None:
        if self._disabled or (self._task and not self._task.done()):
            return
        now = time.monotonic()
        best: dict = {}
        for d in detections:
            if not d.get("box"):  # mock/fallback detections have no real crop
                continue
            label = d["label"]
            if now - self._last_saved.get(label, 0.0) < ZONE_EMBED_INTERVAL_S:
                continue
            if label not in best or d["confidence"] > best[label]["confidence"]:
                best[label] = d
        if not best:
            return
        for label in best:
            self._last_saved[label] = now
        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(
            None, _persist_detections, frame, list(best.values()),
            self.zone_id, self.camera_id, self.engine_key, self.owner_id,
        )
        self._task.add_done_callback(self._on_done)

    def _on_done(self, fut: asyncio.Future) -> None:
        exc = fut.exception()
        if exc:
            # Qdrant/CLIP unavailable: stop trying for this connection rather
            # than logging an error every 5 s. The stream itself keeps working.
            logger.error(f"Persisting detections to Qdrant failed, disabling for this connection: {exc}")
            self._disabled = True


@router.websocket("/ws/vision")
async def vision_websocket(
    websocket: WebSocket,
    engine: str = None,
    zone: str = Query("zone-1", description="Active facility security zone"),
    token: Optional[str] = Query(None, description="JWT — enables persisting detections to Qdrant"),
    camera: Optional[str] = Query(None, description="cameras.id; defaults to the zone's first camera"),
):
    await websocket.accept()

    # Per-connection engine choice (?engine=yolo|finetuned), defaulting to
    # VISION_MODEL_TYPE if omitted — get_engine() caches instances, so
    # switching engines only pays a load cost the first time each one is
    # actually requested; the frontend reconnects with a different ?engine=
    # to switch, no backend restart needed.
    requested = (engine or "").strip().lower()
    if requested and requested not in _VALID_ENGINES:
        logger.warning(f"Unknown engine '{engine}' requested, falling back to default.")
        requested = None
    tracker = get_engine(requested or None)
    # "yolo" and "finetuned" both build a YOLOVisionEngine - type(tracker).__name__
    # can't distinguish them, so report the resolved key instead.
    engine_key = resolve_engine_key(requested or None)

    persister = None
    user = _decode_user(token)
    if ZONE_EMBED_ENABLED and user:
        zone_id, camera_id = await asyncio.to_thread(_resolve_zone_camera, user["id"], zone, camera)
        if zone_id is not None:
            persister = _DetectionPersister(zone_id, camera_id, engine_key, user["id"])

    logger.info(
        f"Client connected to Vision WebSocket stream (engine={engine_key}, zone={zone}, "
        f"persist={'camera=' + persister.camera_id if persister else 'off'})."
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            result = tracker.scan_frame(data)

            # Only boxes the engine already kept (YOLO conf >= 0.5) reach here,
            # i.e. exactly what the stream page displays.
            if persister and result["detections"]:
                persister.offer(data, result["detections"])

            # The frontend renders its own bounding-box overlay from
            # `detections` alone, so `annotated_frame` is never sent.
            await websocket.send_json({
                "status": "success",
                "engine": engine_key,
                "zone": zone,
                "detections": _wire_safe_detections(result["detections"]),
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected from Vision WebSocket stream.")
    except Exception as e:
        logger.error(f"Error in vision websocket loop: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
