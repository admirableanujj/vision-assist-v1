from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from app.engines.vision import VisionTracker

logger = logging.getLogger("vision_routes")
router = APIRouter(tags=["Vision & Object Detection"])

# Single tracker instance for the process lifetime, mirroring audio_routes.py's
# module-level model loading. VisionTracker already resolves to YOLOVisionEngine
# or YOLOEVisionEngine per VISION_MODEL_TYPE (docker-compose already wires this
# env var through — see engines/vision/vision_engine.py), and degrades to
# FallbackVisionEngine internally if the real weights or cv2/ultralytics fail
# to load, so there's no separate None-check needed here the way the old
# inline implementation needed one.
tracker = VisionTracker()

# Placeholder box used only for detections that have none — the mock/fallback
# path (FallbackVisionEngine, used when real weights or cv2/ultralytics aren't
# available) reports "box": None, since the old Streamlit UI only ever drew
# boxes for real detections and showed mock ones as plain text instead. This
# frontend's canvas overlay (useVisionWS.js) has no such text fallback and
# destructures `box` unconditionally, so a None box would throw client-side.
# A fixed placeholder keeps mock detections visibly on-screen (as a clear
# "this is a guess" signal — same spot every time) instead of crashing the
# overlay or silently vanishing.
_MOCK_BOX = [20, 20, 160, 160]

def _wire_safe_detections(detections: list) -> list:
    return [
        {**d, "box": d["box"] or _MOCK_BOX}
        for d in detections
    ]

@router.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"Client connected to Vision WebSocket stream (engine={type(tracker).__name__}).")

    try:
        while True:
            # Receive binary frame (JPEG blob) from frontend. scan_frame() decodes
            # raw bytes itself (see vision_engine.py's _decode_frame), so no local
            # cv2/numpy decoding is needed here.
            data = await websocket.receive_bytes()
            if not data:
                continue

            result = tracker.scan_frame(data)

            # The frontend draws its own bounding-box overlay client-side
            # (useVisionWS.js's renderCanvasOverlay) from the detections list
            # alone — it never asked for a server-rendered annotated_frame, so
            # that half of scan_frame()'s return value is intentionally dropped
            # rather than sent over the wire.
            await websocket.send_json({
                "status": "success",
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
