# Vision Assist

An AI-powered voice+vision Lost & Found assistant — FastAPI backend, React/Vite
frontend, communicating over REST + WebSockets. **Under active development,
not finished** — see [Current Status](#current-status) before assuming
anything below is fully wired.

```
+-----------------------------------------------------------------------------------+
|                                  REACT FRONTEND                                   |
|   (Vite + Tailwind CSS / WebCam Canvas Overlay / WebSocket Audio & Vision)        |
+-----------------------------------------------------------------------------------+
                                   │            ▲
                      REST APIs    │            │  WebSockets
                        (Auth)     │            │  (Live Frames & Bounding Boxes)
                                   ▼            │
+-----------------------------------------------------------------------------------+
|                                 FASTAPI BACKEND                                   |
|  /health   │   /ws/vision   │   /ws/audio   │   /api/v1/auth/login               |
+-----------------------------------------------------------------------------------+
       │                   │                      │
       ▼                   ▼                      ▼
 +-----------+     +---------------+      +---------------------+
 | PostgreSQL|     |  YOLO / YOLOE |      | Whisper STT + Ollama |
 | (Users/Log)     | (Vision Engine)      | (Audio + Intent)     |
 +-----------+     +---------------+      +---------------------+
```

Qdrant and Ollama are also provisioned (`docker-compose.yml`) but not yet
called from any application code except Ollama's own intent-classification
use inside `/ws/audio` — see [Current Status](#current-status).

## Getting Started

```bash
git clone https://github.com/admirableanujj/vision-assist-v1.git
cd vision-assist-v1
cp .env.example .env   # fill in OPENAI_API_KEY if using cloud STT

mkdir -p secrets
printf '%s' 'devpassword123' > secrets/postgres_password.txt
printf '%s' 'unused-for-now' > secrets/qdrant_api_key.txt

docker-compose up -d --build
```

Then open `http://localhost:3000`. **Before you do:** DB migrations currently
fail silently on a fresh database (see Known Issues below) — login will not
work until that's fixed or worked around. For a tested, step-by-step path to
an actually-working local setup (including that workaround) and the same for
GitHub Codespaces, see **`docs/TESTING_LOCAL_AND_CODESPACES.md`**.

## Vision: YOLO / YOLOE

Object detection defaults to YOLO26n (COCO's 80 classes). YOLOE, an
open-vocabulary alternative, can detect additional classes (e.g. `keys`,
`wallet`, `sunglasses` — none of which are COCO classes) at a lower accuracy,
selected via `VISION_MODEL_TYPE=yoloe` in `.env` (container restart required,
no live UI toggle yet). See **`docs/YOLO_VS_YOLOE_GUIDE.md`** for the full
comparison and switching instructions, and
**`docs/FINE_TUNING_YOLO_GUIDE.md`** for the (planning-only) plan to close
that class gap with a real fine-tuned model instead of zero-shot detection.

## Current Status

Several files described in the project structure below are empty stubs, one
route file is dead code, and the database migration script has real bugs.
**`CLAUDE.md`** documents the verified, actual structure, and
**`docs/GAP_ANALYSIS.md`** is a full written audit of what's broken, missing,
or implemented-but-unwired, with a suggested priority order for fixing it.

### Known Issues (see `docs/GAP_ANALYSIS.md` for full detail)

1. **DB migrations fail silently on a fresh database** — a bad `VECTOR` column
   type with no pgvector extension installed, a foreign-key-before-the-table-
   exists ordering bug, and a seed/schema mismatch, all swallowed by a bare
   except. A fresh deploy ends up with zero tables; every login then fails.
   `docs/TESTING_LOCAL_AND_CODESPACES.md` includes a manual workaround.
2. **No `/register` endpoint** despite being planned — only `/login` exists.
3. **CLIP + Qdrant semantic search and the voice engine classes are
   implemented but not wired into any route** (dependencies and Docker
   services are provisioned; nothing calls them yet).
4. Several files are 0-byte stubs: `app/core/config.py`, `app/core/database.py`,
   `app/models/db_models.py`, `app/services/user_service.py`,
   `frontend/src/services/api.js`, and four frontend components. `app/api/router.py`
   is dead code `main.py` never imports.

## Project Structure

```
vision-assist-v1/
├── docker-compose.yml
├── .env.example
├── CLAUDE.md                     <-- verified actual architecture
├── docs/
│   ├── GAP_ANALYSIS.md           <-- full audit: what's broken, missing, or unwired
│   ├── YOLO_VS_YOLOE_GUIDE.md
│   ├── FINE_TUNING_YOLO_GUIDE.md  <-- planning only, nothing implemented
│   └── TESTING_LOCAL_AND_CODESPACES.md
├── secrets/                      <-- git-ignored
│   ├── postgres_password.txt
│   └── qdrant_api_key.txt
├── backend/                      <-- FastAPI / ML service domain
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   <-- entrypoint; includes auth/vision/audio routers directly
│   ├── precheck_db.py            <-- DB migrations + seeding — currently broken, see Known Issues
│   └── app/
│       ├── api/
│       │   ├── router.py         <-- dead code, not used by main.py
│       │   ├── vision_routes.py  <-- /ws/vision, wired to engines/vision (YOLO/YOLOE)
│       │   ├── audio_routes.py   <-- /ws/audio, own inline STT provider dispatch
│       │   ├── auth_routes.py    <-- /api/v1/auth/login only
│       │   └── chat_routes.py    <-- empty
│       ├── core/
│       │   ├── config.py         <-- empty; modules read os.getenv() directly instead
│       │   ├── database.py       <-- empty; auth_routes.py/precheck_db.py each roll their own connection
│       │   ├── security.py       <-- JWT + password verification
│       │   └── logging_config.py
│       ├── engines/               <-- vision, voice, and ML/LLM engines
│       │   ├── vision/            <-- YOLOVisionEngine + YOLOEVisionEngine + fallback — wired
│       │   ├── voice/              <-- not wired (audio_routes.py has its own STT logic)
│       │   └── ml/                 <-- OllamaMLEngine + QueryClassifier (wired); embedding/vector_store (implemented, not wired)
│       ├── services/              <-- empty except __init__.py
│       └── models/
│           ├── db_models.py       <-- empty
│           └── schemas.py         <-- LoginRequest/UserResponse/TokenResponse only
└── frontend/                      <-- React / Vite SPA
    ├── Dockerfile
    ├── naginx.conf                <-- proxies /api and /ws to the backend container
    └── src/
        ├── App.jsx                <-- renders the video/canvas feed directly via useVisionWS()'s refs
        ├── components/
        │   ├── LoginModal.jsx      <-- wired
        │   └── VideoFeed.jsx, ContextPanel.jsx, SystemStatus.jsx, AudioRecorder.jsx  <-- all empty, unreferenced
        ├── hooks/
        │   ├── useVisionWS.js      <-- webcam capture, /ws/vision client, canvas box rendering
        │   └── useAudioWS.js       <-- mic capture, /ws/audio client
        └── services/
            └── api.js              <-- empty; LoginModal.jsx calls fetch() directly instead
```
