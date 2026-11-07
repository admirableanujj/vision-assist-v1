from fastapi import APIRouter, WebSocket, WebSocketDisconnect,Query
import logging

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

@router.websocket("/ws/vision")
async def vision_websocket(
    websocket: WebSocket,
    engine: str = None,
    zone: str = Query("zone-1", description="Active facility security zone"),
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

    logger.info(f"Client connected to Vision WebSocket stream (engine={engine_key}).")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            result = tracker.scan_frame(data)

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
