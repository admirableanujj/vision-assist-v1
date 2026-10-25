from fastapi import APIRouter, WebSocket, WebSocketDisconnect,Query
import logging

from app.engines.vision import VisionTracker

logger = logging.getLogger("vision_routes")
router = APIRouter(tags=["Vision & Object Detection"])

# Resolves to YOLOVisionEngine or YOLOEVisionEngine per VISION_MODEL_TYPE, and
# degrades to FallbackVisionEngine internally on load failure.
tracker = VisionTracker()

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
async def vision_websocket(websocket: WebSocket, zone: str = Query("zone-1", description="Active facility security zone")):
    await websocket.accept()
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
