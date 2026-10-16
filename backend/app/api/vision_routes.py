from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from app.engines.vision import get_engine

logger = logging.getLogger("vision_routes")
router = APIRouter(tags=["Vision & Object Detection"])

_VALID_ENGINES = {"yolo", "yoloe"}

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
async def vision_websocket(websocket: WebSocket, engine: str = None):
    await websocket.accept()

    # Per-connection engine choice (?engine=yolo|yoloe), defaulting to
    # VISION_MODEL_TYPE if omitted — get_engine() caches instances, so
    # switching engines only pays a load cost the first time each one is
    # actually requested; the frontend reconnects with a different ?engine=
    # to switch, no backend restart needed.
    requested = (engine or "").strip().lower()
    if requested and requested not in _VALID_ENGINES:
        logger.warning(f"Unknown engine '{engine}' requested, falling back to default.")
        requested = None
    tracker = get_engine(requested or None)

    logger.info(f"Client connected to Vision WebSocket stream (engine={type(tracker).__name__}).")

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
                "engine": type(tracker).__name__,
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
