# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionAssist is an AI-powered voice+vision Lost & Found assistant: a FastAPI backend + React/Vite frontend communicating over REST + WebSockets, with real-time object detection (YOLO/YOLOE), speech I/O, and LLM-driven intent handling.

**Entry point:** `backend/main.py` — a FastAPI app (`uvicorn main:app`) that runs DB migrations/seeding on startup (`precheck_db.py`) then serves REST + WebSocket routes.

## Development Environment

```bash
# Backend (from backend/)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (from frontend/)
npm install
npm run dev

# Full stack via Docker Compose (from repo root)
docker-compose up -d --build
```

Requires `.env` (copy from `.env.example`) and `./secrets/` (`postgres_password.txt`, `qdrant_api_key.txt`, git-ignored) before `docker-compose up` — see `.env.example` for the exact variables.

Services (`docker-compose.yml`):

| Service | Container | Port | Role |
|---|---|---|---|
| `backend` | `vision_assist_api` | 8000 | FastAPI app |
| `frontend` | `vision_assist_web` | 3000 | React/Vite, nginx-served |
| `postgres` | `vision_assist_db` | 5432 | User auth + item persistence |
| `ollama` | `vision_assist_llm_local` | 11434 | Local LLM (llama3) |
| `qdrant` | `vision_assist_qdrant` | 6333/6334 | Vector DB for semantic item search |

## Testing

`requirements.txt` includes `pytest`/`pytest-cov`/`httpx` for backend testing; no suite has been written yet.

## Architecture

### Repository Layout

```
vision-assist-v1/
├── docker-compose.yml
├── .env.example
├── backend/                          # FastAPI / ML service domain
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                       # Entrypoint — wires auth/vision/audio routers
│   ├── precheck_db.py                # DB migrations + first-boot seeding
│   └── app/
│       ├── api/                      # HTTP + WebSocket routes
│       │   ├── vision_routes.py      # /ws/vision — YOLO/YOLOE detection
│       │   ├── audio_routes.py       # /ws/audio — STT + intent routing
│       │   └── auth_routes.py        # /api/v1/auth/login
│       ├── core/                     # Security, logging
│       ├── engines/                  # Vision, voice, and ML/LLM engines
│       │   ├── vision/               # YOLOVisionEngine + YOLOEVisionEngine + fallback
│       │   ├── voice/                # Whisper STT + TTS
│       │   └── ml/                   # Ollama, query classification, CLIP/Qdrant embeddings
│       └── models/                   # Pydantic schemas
└── frontend/                         # React / Vite SPA
    └── src/
        ├── App.jsx                   # Main dashboard
        ├── components/LoginModal.jsx
        └── hooks/
            ├── useVisionWS.js        # Webcam capture, /ws/vision client, canvas box rendering
            └── useAudioWS.js         # Mic capture, /ws/audio client
```

### Request Flow

```
Vision: browser webcam → useVisionWS.js captures a 640x480 JPEG every ~100ms
  → WS /ws/vision?engine=yolo|yoloe&zone=... (raw JPEG bytes) → vision_routes.py
  → get_engine(engine).scan_frame(bytes)     # cached per engine, degrades to mock on failure
  → {"status": "success", "engine": "...", "zone": "...", "detections": [{label, confidence, box}, ...]}
  → useVisionWS.js draws boxes on its own <canvas> — the server never renders an annotated image
  (CameraFeedDashboard.jsx's YOLO/YOLOE toggle changes `engine` and reconnects the socket — no restart)

Audio: browser mic → useAudioWS.js → WS /ws/audio (webm blob) → audio_routes.py
  → transcribe via STT_PROVIDER (faster-whisper default, or whisper/openai)
  → QueryClassifier.classify() (Ollama llama3, JSON intent: locate/note/alarm/general)
  → canned/Ollama response per intent → {"status", "intent", "transcript", "response"}

Auth: LoginModal.jsx → fetch POST /api/v1/auth/login → auth_routes.py → JWT via core/security.py
```

### Key Design Patterns

- **Vision engine (`app/engines/vision/vision_engine.py`)**: `BaseVisionEngine` ABC → `YOLOVisionEngine` / `YOLOEVisionEngine` (share inference via `_UltralyticsScanMixin`, only model loading differs) → `FallbackVisionEngine` mock degrade path. `get_engine(engine_type)` lazily constructs and caches one instance per type — both YOLO and YOLOE can be resident at once, so `vision_routes.py` picks per-connection via `/ws/vision`'s `?engine=` query param (falling back to `VISION_MODEL_TYPE` if omitted) without restarting anything; `docker-compose.yml` wires `YOLO_MODEL_PATH`/`FINE_TUNED_YOLO_MODEL_PATH`/`VISION_MODEL_TYPE`/`VISION_CUSTOM_CLASSES` through for the startup default. `scan_frame(image_buffer)` returns `{"detections": [...], "annotated_frame": ...}` — `vision_routes.py` only forwards `detections` (plus which `engine` served them), since the client renders its own overlay.
- **`CameraFeedDashboard.jsx`'s YOLO/YOLOE toggle drives `useVisionWS`'s `engine` param** (alongside the existing `zoneId` param), which is just a dependency of the effect that opens the `/ws/vision` socket — changing either triggers a reconnect. `activeEngine` (set from each response's `engine` field) only reflects what the server actually confirmed, so the UI can show "switching…" during the gap rather than assuming the click took effect immediately.
- **`FINE_TUNED_YOLO_MODEL_PATH` overrides `YOLOVisionEngine`'s weights only — never `YOLOEVisionEngine`'s.** `YOLO_MODEL_PATH` is a single var shared by both, which was harmless while only one engine was ever resident — now that `get_engine()` can hold both at once, pointing the shared var at a fine-tuned checkpoint would make `YOLOEVisionEngine` try to load it too (wrong architecture) and silently degrade to mock detections. `YOLO_MODEL_PATH` still works as a legacy fallback read by both; new code pointing at a fine-tuned YOLO checkpoint should use the new var instead.
- **Mock/fallback detections report `box: None`** — `vision_routes.py` substitutes a fixed placeholder box before sending, since the frontend's canvas renderer destructures `box` unconditionally with no null check.
- **Audio provider selection (`app/api/audio_routes.py`)**: `STT_PROVIDER` env var, loaded once at module level with the model left `None` on failure rather than crashing — match this functional, env-var-dispatch idiom for new engine selection logic, except where the heavier ABC pattern already exists (`engines/vision/`).
- **CLIP + Qdrant semantic search (`app/engines/ml/vector_store.py` + `embedding_engine.py`)** and **the voice engine classes (`app/engines/voice/`)** are implemented but not yet wired into a route — `audio_routes.py` currently has its own separate inline STT dispatch instead of using `engines/voice/`.
- **Model weights are baked into `/opt/weights/`, not a relative path.** `docker-compose.yml` bind-mounts `./backend` over `/workspace` for hot-reload, which would otherwise hide anything the Dockerfile downloaded there. `DEFAULT_LOCAL_WEIGHTS`/`DEFAULT_YOLOE_WEIGHTS` in `vision_engine.py` point at the baked-in absolute paths; `YOLO_MODEL_PATH` still overrides both as before.
- **Backend↔Postgres and frontend↔backend connect via `host.docker.internal` (mapped to the bridge gateway through `extra_hosts: host-gateway`), not the service hostname.** Direct container-to-container bridge routing was observed broken in a constrained environment (GitHub Codespaces) while the published-port path through the host worked; this routes through the latter unconditionally rather than depending on the former being fixed. If you ever "simplify" `POSTGRES_HOST` or nginx's `proxy_pass` back to the plain service name, know why it was changed first.

See `docs/YOLO_VS_YOLOE_GUIDE.md` for choosing/configuring the two detection engines, and `docs/FINE_TUNING_YOLO_GUIDE.md` for the (planning-only, not yet implemented) plan to close YOLO's class gap on `keys`/`wallet`/`sunglasses`.
