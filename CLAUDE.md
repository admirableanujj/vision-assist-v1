# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionAssist is an AI-powered voice+vision Lost & Found assistant: a FastAPI backend + React/Vite frontend communicating over REST + WebSockets, with real-time object detection (YOLO/YOLOE), speech I/O, and LLM-driven intent handling.

**Entry point:** `backend/main.py` — a FastAPI app (`uvicorn main:app`) that runs DB migrations/seeding on startup (`precheck_db.py`) then serves REST + WebSocket routes. `precheck_db.py` creates the full schema if missing and, only when `users` is genuinely empty, seeds `test_user`/`test_user` and `admin`/`adminpassword123` — no manual DB setup is needed to log in.

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

Requires `.env` (copy from `.env.example`) and `./secrets/` (`postgres_password`, `qdrant_api_key`, git-ignored; UTF-8, no BOM, no trailing newline — on Windows write them with `[IO.File]::WriteAllText`, never `echo >`) before `docker-compose up` — see `.env.example` for the exact variables.

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
│   ├── main.py                       # Entrypoint — wires auth/vision/audio/zone routers
│   ├── precheck_db.py                # DB migrations + first-boot seeding (see below)
│   └── app/
│       ├── api/                      # HTTP + WebSocket routes
│       │   ├── vision_routes.py      # /ws/vision — YOLO/YOLOE detection
│       │   ├── audio_routes.py       # /ws/audio — STT + intent routing
│       │   ├── auth_routes.py        # /api/v1/auth/login
│       │   └── zone_routes.py        # /api/v1/zones, /items — zone inventory + item registration
│       ├── core/                     # security.py, database.py, logging_config.py
│       ├── engines/                  # Vision, voice, and ML/LLM engines
│       │   ├── vision/               # YOLOVisionEngine + YOLOEVisionEngine + fallback
│       │   ├── voice/                # Whisper STT + TTS
│       │   └── ml/                   # Ollama, query classification, CLIP/Qdrant embeddings
│       └── models/                   # Pydantic schemas (db_models.py, schemas.py)
└── frontend/                         # React / Vite SPA
    └── src/
        ├── App.jsx                   # Auth gate; routes to ZoneInventoryDashboard or CameraFeedDashboard
        ├── pages/
        │   ├── ZoneInventoryDashboard.jsx  # Default view post-login — zones, registered items
        │   └── CameraFeedDashboard.jsx     # Live vision feed + YOLO/YOLOE toggle, reached via "Camera Stream"
        ├── components/LoginModal.jsx
        └── hooks/
            ├── useVisionWS.js        # Webcam capture, /ws/vision client, canvas box rendering
            └── useAudioWS.js         # Mic capture, /ws/audio client
```

`backend/app/api/router.py` and `chat_routes.py`, `backend/app/core/config.py`, `backend/app/services/`,
and `frontend/src/App_bkp.jsx` plus `frontend/src/components/{VideoFeed,AudioRecorder,SystemStatus,ContextPanel}.jsx`
and `frontend/src/services/api.js` all exist but are dead — empty, or never imported by `main.py`/`App.jsx`.
Don't treat their presence as documentation of intended behavior.

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

Zones/items: App.jsx routes a logged-in user to ZoneInventoryDashboard.jsx by
default → GET /api/v1/zones, /api/v1/items/registered (zone_routes.py, scoped
to the authenticated user). Clicking "Camera Stream" switches App.jsx's
activeView to CameraFeedDashboard.jsx (the vision flow above); "Zone
Inventory" in its header switches back.
```

### Key Design Patterns

- **Vision engine (`app/engines/vision/vision_engine.py`)**: `BaseVisionEngine` ABC → `YOLOVisionEngine` / `YOLOEVisionEngine` (share inference via `_UltralyticsScanMixin`, only model loading differs) → `FallbackVisionEngine` mock degrade path. **App-facing choice is deliberately just two engine keys, `"yolo"` (base, COCO-pretrained) and `"finetuned"` (this project's 48-class lostfound checkpoint) — both build a `YOLOVisionEngine`, just with different weights, so a user can directly compare base vs. fine-tuned detection on the same scene.** `"yoloe"` still exists as a third key (config-only, via `VISION_MODEL_TYPE=yoloe`), just not exposed in `_VALID_ENGINES`/the frontend toggle. `get_engine(engine_type)` lazily constructs and caches one instance per key via dedicated builder functions (`_build_yolo_engine`/`_build_finetuned_engine`/`_build_yoloe_engine`) — all three can be resident at once, so `vision_routes.py` picks per-connection via `/ws/vision`'s `?engine=` query param (falling back to `VISION_MODEL_TYPE` if omitted) without restarting anything; `docker-compose.yml` wires `YOLO_MODEL_PATH`/`FINE_TUNED_YOLO_MODEL_PATH`/`VISION_MODEL_TYPE`/`VISION_CUSTOM_CLASSES` through for the startup default. `scan_frame(image_buffer)` returns `{"detections": [...], "annotated_frame": ...}` — `vision_routes.py` only forwards `detections` (plus `resolve_engine_key()`'s resolved key, not `type(tracker).__name__`, since "yolo"/"finetuned" are otherwise indistinguishable — both are a `YOLOVisionEngine`), since the client renders its own overlay.
- **`CameraFeedDashboard.jsx`'s YOLO/Fine-tuned toggle drives `useVisionWS`'s `engine` param** (alongside the existing `zoneId` param), which is just a dependency of the effect that opens the `/ws/vision` socket — changing either triggers a reconnect. `activeEngine` (set from each response's `engine` field) only reflects what the server actually confirmed, so the UI can show "switching…" during the gap rather than assuming the click took effect immediately.
- **`FINE_TUNED_YOLO_MODEL_PATH` only overrides the `"finetuned"` key's weights, never `"yolo"`'s — the two are deliberately independent, not a shared-then-swapped var.** An earlier design had `YOLOVisionEngine`'s own constructor read `FINE_TUNED_YOLO_MODEL_PATH` ahead of `YOLO_MODEL_PATH` whenever called with no explicit `weights_path`, which worked for a single always-resident engine but would make `"yolo"` silently *become* the fine-tuned model the moment that var is set — leaving nothing to compare it against once both are independently selectable. `get_engine()`'s builders now pass `weights_path` explicitly for both keys so neither can be affected by the other's config. (`YOLOVisionEngine.__init__`'s own fallback-chain resolution is unchanged, for any caller that still constructs it directly with no args.) `YOLO_MODEL_PATH` still works as a legacy override for the `"yolo"` key specifically.
- **Mock/fallback detections report `box: None`** — `vision_routes.py` substitutes a fixed placeholder box before sending, since the frontend's canvas renderer destructures `box` unconditionally with no null check.
- **Audio provider selection (`app/api/audio_routes.py`)**: `STT_PROVIDER` env var, loaded once at module level with the model left `None` on failure rather than crashing — match this functional, env-var-dispatch idiom for new engine selection logic, except where the heavier ABC pattern already exists (`engines/vision/`).
- **CLIP + Qdrant semantic search (`app/engines/ml/vector_store.py` + `embedding_engine.py`)** and **the voice engine classes (`app/engines/voice/`)** are implemented but not yet wired into a route — `audio_routes.py` currently has its own separate inline STT dispatch instead of using `engines/voice/`.
- **Model weights are baked into `/opt/weights/`, not a relative path.** `docker-compose.yml` bind-mounts `./backend` over `/workspace` for hot-reload, which would otherwise hide anything the Dockerfile downloaded there. `DEFAULT_LOCAL_WEIGHTS`/`DEFAULT_YOLOE_WEIGHTS` in `vision_engine.py` point at the baked-in absolute paths; `YOLO_MODEL_PATH` still overrides both as before.
- **Frontend↔backend connects via `host.docker.internal` (mapped to the bridge gateway through `extra_hosts: host-gateway`), hardcoded in `nginx.conf`'s `proxy_pass` — not the service hostname.** Direct container-to-container bridge routing was observed broken in a constrained environment (GitHub Codespaces) while the published-port path through the host worked. Don't "simplify" this back to the plain service name without knowing why it was changed.
- **Backend↔Postgres is configurable via the `POSTGRES_HOST` env var** (`.env.example` default: `vision_assist_db`, the plain service hostname — works on ordinary Docker). **In GitHub Codespaces, override it to `host.docker.internal`** for the same bridge-routing reason as above — see `docs/TESTING_LOCAL_AND_CODESPACES.md`'s Codespaces quick start. Unlike the frontend/nginx case, this one is deliberately an env var rather than hardcoded, since hardcoding `host.docker.internal` unconditionally broke environments where the plain service hostname works fine.

See `docs/YOLO_VS_YOLOE_GUIDE.md` for background on the two Ultralytics engine types, and `docs/FINE_TUNING_YOLO_GUIDE.md` for the full fine-tuning writeup — a YOLO26n scoped to a curated 48-class home/garage lost-and-found vocabulary (20 of COCO's 80 classes kept + 28 new ones like `keys`/`wallet`/`glasses`/`screwdriver`). **This has actually been run** (`finetuning/kaggle/RUN_LOG.md` has the full record): three training variants were compared (baseline, a bigger-COCO-sample variant, an unfrozen-backbone variant), and the bigger-COCO-sample run ("run 2" — `--coco-sample-size 5000`, `freeze=10`, `cls_pw=0.5`) was chosen and is the checkpoint now baked into `backend/weights/lostfound_yolo26n_best.pt` / `/opt/weights/lostfound_yolo26n_best.pt` — it improved the catastrophic-forgetting check on every one of the 20 kept COCO classes vs. the baseline, for only a marginal novel-class cost. The real three-way golden-set benchmark (`finetuning/evaluate.py`, base YOLO vs. YOLOE vs. this fine-tuned model on self-captured photos) is still pending — needs the golden set captured/annotated first (`finetuning/README.md` Step 4).
