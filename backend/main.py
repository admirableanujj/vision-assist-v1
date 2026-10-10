# import async_timeout
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.vision_routes import router as vision_router
from app.api.audio_routes import router as audio_router
from precheck_db import run_migrations_and_seeding
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()    
app = FastAPI(title="Vision Assist API", version="2.0.0")

# Enable CORS for React dev server & container proxying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    run_migrations_and_seeding()

# Include all modular routers
app.include_router(auth_router)
app.include_router(vision_router)
app.include_router(audio_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "services": {"postgres": "connected", "qdrant": "healthy"}}

@app.get("/")
def read_root():
    return {
        "service": "Vision Assist API",
        "status": "online",
        "docs": "/docs",
        "message": "Welcome to the Vision Assist API!"
    }
    
