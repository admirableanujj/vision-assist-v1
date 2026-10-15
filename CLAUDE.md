# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionAssist is an AI-powered voice+vision Lost & Found assistant. This repo (`vision-assist-v1`) is a **full rewrite** of an earlier Streamlit prototype (`vision-assist`) into a FastAPI backend + React/Vite frontend, communicating over REST + WebSockets instead of Streamlit's rerun-the-whole-script model.

**This rewrite is mid-migration, not finished.** Several pieces the `README` describes as already in place are actually empty stub files or were ported from the old repo without being adapted to this one's structure. Don't trust the README's file-tree description as ground truth — verify against the actual file before relying on it. See `docs/GAP_ANALYSIS.md` for the full, verified inventory of what's real vs. aspirational as of this writing.

**Entry point:** `backend/main.py` — a FastAPI app (`uvicorn main:app`) that runs DB migrations/seeding on startup (`precheck_db.py`) then serves REST + WebSocket routes. **Migrations are currently broken** (see Known Issues #1) — don't assume a fresh container has a working database.

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
| `frontend` | (check compose for name) | — | React/Vite, nginx-served |
| `postgres` | `vision_assist_db` | — | User auth + item persistence |
| `ollama` | `vision_assist_llm_local` | 11434 | Local LLM (llama3) |
| `qdrant` | — | 6333/6334 | Vector DB — client/deps present, **not called from any application code** |

## Testing

**No test suite exists yet.** `requirements.txt` lists `pytest`/`pytest-cov`/`httpx`, but there is no `tests/` directory anywhere in `backend/`. Anyone adding tests is starting from zero, not extending something broken.

## Architecture

### Repository Layout (verified, not the README's version)

```
vision-assist-v1/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                       # FastAPI entrypoint — includes auth/vision/audio routers directly
│   ├── precheck_db.py                # DB migrations + seeding — currently broken, see Known Issues #1
│   └── app/
│       ├── api/
│       │   ├── router.py             # DEAD CODE — builds its own unused FastAPI() app, not imported anywhere
│       │   ├── vision_routes.py       # /ws/vision — now wired to engines/vision (YOLO/YOLOE)
│       │   ├── audio_routes.py        # /ws/audio — STT provider dispatch (faster-whisper/whisper/openai) + Ollama intent routing
│       │   ├── auth_routes.py         # POST /api/v1/auth/login only — no /register despite README
│       │   └── chat_routes.py        # EMPTY FILE
│       ├── core/
│       │   ├── config.py             # EMPTY FILE — no centralized settings; every module reads os.getenv() itself
│       │   ├── database.py           # EMPTY FILE — auth_routes.py and precheck_db.py each hand-roll their own psycopg2 connection
│       │   ├── security.py           # JWT + password verification — functional
│       │   └── logging_config.py     # get_logger() factory — the correct import path for every module under app/
│       ├── engines/                   # Core ML/AI logic, ported from the old repo's top-level ml_engine/voice_engine/vision_engine
│       │   ├── vision/                # YOLOVisionEngine + YOLOEVisionEngine + FallbackVisionEngine — see Key Design Patterns
│       │   ├── voice/                 # Whisper STT + TTS classes — ported but NOT wired into audio_routes.py (which has its own inline STT dispatch instead)
│       │   └── ml/                    # OllamaMLEngine, QueryClassifier (wired, used by audio_routes.py), embedding_engine.py + vector_store.py (ported, not wired to anything)
│       ├── services/                  # EMPTY except __init__.py — README's documented user_service.py does not exist; auth_routes.py inlines raw SQL instead
│       └── models/
│           ├── db_models.py          # EMPTY FILE — no ORM models exist
│           └── schemas.py            # LoginRequest/UserResponse/TokenResponse only — no RegisterRequest
└── frontend/
    └── src/
        ├── App.jsx                    # Main dashboard — renders video/canvas directly via useVisionWS()'s refs
        ├── App_bkp.jsx                 # Older, less complete version of App.jsx — superseded, not imported
        ├── components/
        │   ├── LoginModal.jsx          # Complete, wired — calls /api/v1/auth/login directly via fetch()
        │   ├── VideoFeed.jsx            # EMPTY — App.jsx bypasses it, renders the feed inline instead
        │   ├── ContextPanel.jsx         # EMPTY, unreferenced
        │   ├── SystemStatus.jsx         # EMPTY, unreferenced
        │   └── AudioRecorder.jsx        # EMPTY, unreferenced — its job is fully absorbed by hooks/useAudioWS.js
        ├── hooks/
        │   ├── useVisionWS.js           # Complete — owns webcam capture, the /ws/vision WebSocket, AND client-side canvas box rendering
        │   └── useAudioWS.js            # Complete — owns mic capture and the /ws/audio WebSocket
        └── services/
            └── api.js                   # EMPTY — no centralized API client; LoginModal.jsx calls fetch() directly instead
```

### Request Flow

```
Vision: browser webcam → useVisionWS.js captures a 640x480 JPEG every ~100ms
  → WS /ws/vision (raw JPEG bytes) → vision_routes.py
  → VisionTracker.scan_frame(bytes)         # YOLO or YOLOE per VISION_MODEL_TYPE, degrades to mock on failure
  → {"status": "success", "detections": [{label, confidence, box}, ...]}
  → useVisionWS.js draws boxes on its own <canvas> — the server never renders an annotated image

Audio: browser mic → useAudioWS.js → WS /ws/audio (webm blob) → audio_routes.py
  → transcribe via STT_PROVIDER (faster-whisper default, or whisper/openai)
  → QueryClassifier.classify() (Ollama llama3, JSON intent: locate/note/alarm/general)
  → canned/Ollama response per intent → {"status", "intent", "transcript", "response"}

Auth: LoginModal.jsx → fetch POST /api/v1/auth/login → auth_routes.py (raw psycopg2 query) → JWT via core/security.py
```

### Key Design Patterns

- **Vision engine (`app/engines/vision/vision_engine.py`)**: `BaseVisionEngine` ABC → `YOLOVisionEngine` / `YOLOEVisionEngine` (share inference logic via `_UltralyticsScanMixin`, only model loading differs) → `FallbackVisionEngine` mock degrade path. `VisionTracker` (resolved once at module import) picks YOLO vs. YOLOE from the `VISION_MODEL_TYPE` env var (`"yolo"` default, or `"yoloe"`) — docker-compose already wires `YOLO_MODEL_PATH`/`VISION_MODEL_TYPE`/`VISION_CUSTOM_CLASSES` through to the container. `scan_frame(image_buffer)` accepts raw bytes directly (decodes internally) and returns `{"detections": [...], "annotated_frame": ...}` — `vision_routes.py` only forwards `detections` to the frontend, since the client renders its own overlay.
- **Mock/fallback detections report `box: None`** (there's nothing to draw a real box around) — `vision_routes.py` substitutes a fixed placeholder box before sending over the wire, since the frontend's canvas renderer destructures `box` unconditionally and has no text-only fallback path (unlike the old Streamlit UI, which showed mock detections as text instead of a box).
- **Audio provider selection (`app/api/audio_routes.py`)** follows the same env-var-driven pattern: `STT_PROVIDER` (`faster-whisper` default, or `whisper`/`openai`), loaded once at module level with a try/except that leaves the model `None` on failure rather than crashing — **this is the idiom to match** when extending any engine selection logic in this repo (functional, env-var dispatch — not the old repo's heavier ABC-per-engine pattern, except where that pattern already existed, as in `engines/vision/`).
- **`app/engines/ml/vector_store.py` + `embedding_engine.py` (CLIP + Qdrant semantic search)** were ported from the old repo but are **not wired into any route** — `requirements.txt` already has `qdrant-client`/`sentence-transformers`/the CLIP git dependency, and `docker-compose.yml` already provisions a `qdrant` service, but nothing in `app/api/` calls either module. Out of scope for the YOLO/YOLOE integration pass that produced this file — see `docs/GAP_ANALYSIS.md`.
- **`app/engines/voice/`** (ported `voice_stt.py`/`voice_tts.py`, class-based like the old repo) is similarly unwired — `audio_routes.py` has its own separate, simpler inline STT dispatch instead. Not touched by this pass either.

## Known Issues

1. **DB migrations are broken and fail silently.** `precheck_db.py`'s `items` table declares `embedding VECTOR(512)` with no pgvector extension installed anywhere (`docker-compose.yml` uses plain `postgres:16-alpine`) — this `CREATE TABLE` fails outright. Separately, the `cameras` table is created with a foreign key to `zones` *before* the `zones` table exists later in the same script, and the `zones` seed insert's columns don't match `zones`' actual schema. All of this is wrapped in a single try/except that only logs + rolls back, never re-raises — so `main.py`'s startup proceeds as if migrations succeeded, and a fresh database ends up with **zero tables**, failing every subsequent login with "relation does not exist." Not fixed by this pass (out of scope — not vision-related) but this blocks the whole app from working end-to-end on a fresh deploy; fix before any demo.
2. **`app/api/router.py` is dead code** — instantiates its own unused `FastAPI()` app and only includes `audio_router`; `main.py` never imports it and wires its three routers directly instead. Safe to delete or to actually consolidate routing through it — just pick one.
3. **Several files the README describes as existing do not**: `app/core/config.py`, `app/core/database.py`, `app/models/db_models.py`, `app/services/user_service.py`, `frontend/src/services/api.js`, and four frontend components (`VideoFeed.jsx`, `ContextPanel.jsx`, `SystemStatus.jsx`, `AudioRecorder.jsx`) are all 0-byte stubs. `auth_routes.py` works around the missing `database.py`/`user_service.py`/`config.py` by inlining its own `psycopg2` connection and raw queries directly.
4. **No `/register` endpoint** despite the README documenting one — `auth_routes.py` only implements `/api/v1/auth/login`, and `schemas.py` has no `RegisterRequest`.
5. **CLIP/Qdrant semantic search is provisioned but unwired** (dependencies, Docker service, and the ported `embedding_engine.py`/`vector_store.py` all exist; nothing calls them). Same for the ported `voice/` engine classes vs. `audio_routes.py`'s own separate inline STT logic. Both ported-but-duplicate situations are the same shape as the vision engine was before this pass — worth the same treatment (wire the real module, delete the duplicate) if either becomes a priority.

See `docs/GAP_ANALYSIS.md` for the full, evidence-cited version of all of the above, `docs/YOLO_VS_YOLOE_GUIDE.md` for how to choose/configure the two detection engines, and `docs/FINE_TUNING_YOLO_GUIDE.md` for the (planning-only, not yet implemented) plan to close YOLO's class gap on `keys`/`wallet`/`sunglasses`.
