import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import logging

logger = logging.getLogger("vision_routes")
router = APIRouter(tags=["Vision & Object Detection"])

# Load YOLO model (Using YOLOv8 nano / small for fast edge inference)
# The model will download automatically or load from local weights if pre-cached
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    logger.warning(f"Could not load local YOLO weights, falling back: {e}")
    model = None

@router.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to Vision WebSocket stream.")

    try:
        while True:
            # Receive binary frame (JPEG blob) from frontend
            data = await websocket.receive_bytes()

            if not data:
                continue

            # Convert binary data to OpenCV image matrix
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            detections = []

            if model is not None:
                # Run YOLO inference on the frame
                results = model(frame, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Extract coordinates [x1, y1, x2, y2]
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]

                        detections.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "box": [round(c, 1) for c in coords] # [x1, y1, x2, y2]
                        })

            # Send JSON detection payload back to React client
            await websocket.send_json({
                "status": "success",
                "detections": detections
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected from Vision WebSocket stream.")
    except Exception as e:
        logger.error(f"Error in vision websocket loop: {e}")
        try:
            await websocket.close()
        except:
            pass