from app.api.audio_routes import router as audio_router
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Vision Assist API", version="2.0.0")
app.include_router(audio_router)