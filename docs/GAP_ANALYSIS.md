# Gap Analysis: Known Issues and Incomplete Areas

Verified by reading the code, not by trusting the README. Split into what the
YOLO/YOLOE integration fixed, what's related but explicitly out of scope, and
what's unrelated but serious enough to flag regardless.

## 1. YOLO/YOLOE — what was wrong, and what's fixed

The vision engine (`BaseVisionEngine` → `YOLOVisionEngine`/`YOLOEVisionEngine`
→ `FallbackVisionEngine`, shared inference via `_UltralyticsScanMixin`) was
implemented correctly in `backend/app/engines/vision/vision_engine.py`, but
was never actually integrated:

| Problem | Evidence | Fixed? |
|---|---|---|
| Broken import — `vision_engine.py` imported `from logging_config import get_logger`, a path that doesn't resolve in this repo's layout (the logger lives at `app.core.logging_config`). The module failed to import at all. | `vision_engine.py:27` (before fix) | ✅ Yes |
| `vision_routes.py` never used the engine module — ran its own separate, cruder inline `YOLO("yolov8n.pt")` call instead: no YOLOE, no confidence threshold, no fallback, an older/less accurate checkpoint than `vision_engine.py`'s own `yolo26n.pt` default. | `vision_routes.py` (original, see git history) | ✅ Yes — now imports and uses `VisionTracker` |
| `docker-compose.yml`'s `backend` service already wires `YOLO_MODEL_PATH`/`VISION_MODEL_TYPE`/`VISION_CUSTOM_CLASSES` through to the container, but nothing consumed them, since the route bypassed the engine module that reads them. | `docker-compose.yml:30-32` vs. the pre-fix `vision_routes.py` | ✅ Fixed as a side effect of wiring the real engine |
| `scan_frame()`'s return contract (`{"detections": [...], "annotated_frame": ...}`) doesn't match what this frontend needs — `useVisionWS.js` renders its own canvas overlay client-side from `detections` alone and never asked for a server-rendered image. | `useVisionWS.js` (read in full) | ✅ Handled — route only forwards `detections`, never sends `annotated_frame` |
| **New bug this wiring would have introduced**: `FallbackVisionEngine` reports `"box": None` per mock detection; this frontend's canvas code destructures `box` unconditionally with no null check — would throw client-side the moment the fallback path engaged. | `vision_engine.py`'s `FallbackVisionEngine.scan_frame()` + `useVisionWS.js`'s `renderCanvasOverlay` | ✅ Fixed at the WebSocket boundary (`vision_routes.py`'s `_wire_safe_detections`) — doesn't touch the shared engine class or any frontend file |
| Both model checkpoints (and YOLOE's CLIP dependency) would resolve at first real runtime use instead of build time — a live `pip install` inside an already-running process, which doesn't take effect without an interpreter restart and can crash it outright. | `backend/Dockerfile` (had no pre-download step) | ✅ Fixed — both checkpoints pre-baked, YOLOE's `set_classes()` exercised at build time |

**Net effect**: `/ws/vision` now actually runs real YOLO or YOLOE (per
`VISION_MODEL_TYPE`), with the fallback/degrade path working safely, and the
Docker image avoids a known crash class entirely. `docs/YOLO_VS_YOLOE_GUIDE.md`
documents how to use/switch engines in this repo's actual (env-var + container
restart, not live UI toggle) reality. `docs/FINE_TUNING_YOLO_GUIDE.md` is the
plan — datasets, classes, mechanics, a researched LoRA/QLoRA verdict (don't
use it — see that doc's Section 5), and a three-way evaluation methodology —
as **planning only**: no training code was written or run.

## 2. Explicitly out of scope (flagged, not fixed)

- **CLIP + Qdrant semantic search.** `app/engines/ml/embedding_engine.py` and
  `vector_store.py` are implemented, and `requirements.txt`/`docker-compose.yml`
  already provision everything needed (CLIP git dependency,
  `sentence-transformers`, `qdrant-client`, a `qdrant` service) — but nothing
  in `app/api/` calls either module. Same shape of problem vision was in
  before being wired up; same fix would apply (wire it into a route) if/when
  this becomes a priority.
- **`app/engines/voice/` (`voice_stt.py`/`voice_tts.py`, class-based).**
  `audio_routes.py` has its own separate, simpler inline STT provider dispatch
  instead of using these. Not touched here.

## 3. Unrelated, but serious enough to flag regardless

None of this is vision-related and none of it was touched by the YOLO/YOLOE
integration — listed here so it doesn't get lost, roughly in priority order.

### 3.1 DB migrations are broken and fail silently (blocks the whole app)

`precheck_db.py`:
- The `items` table declares `embedding VECTOR(512)` with no pgvector
  extension installed anywhere — `docker-compose.yml` uses plain
  `postgres:16-alpine`. This `CREATE TABLE` fails outright.
- The `cameras` table is created with a foreign key to `zones` *before* the
  `zones` table exists later in the same script — a second, independent
  failure.
- The `zones` seed insert's columns (`camera_id, name, x1, y1, x2, y2`) don't
  match `zones`' actual schema (`id, user_id, name`).
- All of this runs inside one try/except that only logs + rolls back, never
  re-raises. `main.py`'s startup proceeds as if migrations succeeded. **A
  fresh database ends up with zero tables**, and every subsequent login fails
  with "relation does not exist."

This blocks the entire app from working end-to-end on a clean deploy — fix
before any demo.

### 3.2 Dead/missing code, by file

| File | Status |
|---|---|
| `backend/app/api/router.py` | Dead code — builds its own unused `FastAPI()` app, includes only `audio_router`; `main.py` never imports it and wires its three routers directly instead |
| `backend/app/api/chat_routes.py` | 0 bytes |
| `backend/app/core/config.py` | 0 bytes — no centralized settings; every module reads `os.getenv()` itself |
| `backend/app/core/database.py` | 0 bytes — `auth_routes.py` and `precheck_db.py` each hand-roll their own `psycopg2` connection instead |
| `backend/app/models/db_models.py` | 0 bytes — no ORM models exist anywhere |
| `backend/app/services/user_service.py` | **Does not exist at all** (README.md documents it as replacing `UserManager`) — `auth_routes.py` inlines raw SQL instead |
| `frontend/src/services/api.js` | 0 bytes — no centralized API client; `LoginModal.jsx` calls `fetch()` directly instead |
| `frontend/src/components/VideoFeed.jsx` | 0 bytes, unreferenced — `App.jsx` renders the video/canvas feed inline via `useVisionWS()`'s refs instead |
| `frontend/src/components/ContextPanel.jsx` | 0 bytes, unreferenced |
| `frontend/src/components/SystemStatus.jsx` | 0 bytes, unreferenced |
| `frontend/src/components/AudioRecorder.jsx` | 0 bytes, unreferenced — fully absorbed by `hooks/useAudioWS.js` |
| `frontend/src/App_bkp.jsx` | Older, less complete version of `App.jsx` (hand-rolled webcam/WebSocket logic later extracted into `useVisionWS.js`; no audio feature at all) — superseded, not imported anywhere |

### 3.3 Auth is incomplete

`auth_routes.py` implements `POST /api/v1/auth/login` only — no `/register`
endpoint exists anywhere (not in the router, not in `schemas.py` as a
`RegisterRequest`, not in `security.py` as a password-hash-on-create helper),
despite README.md documenting one.

### 3.4 No tests

`requirements.txt` lists `pytest`/`pytest-cov`/`httpx`, but there is no
`tests/` directory anywhere in `backend/`. Starting from zero, not extending
something broken.

## 4. Suggested priority order

1. **Fix DB migrations** (3.1) — nothing else matters if the app can't boot
   with a working database.
2. **Decide on CLIP/Qdrant and voice engine wiring** (Section 2) — same
   implemented-but-unwired shape just fixed for vision; same fix pattern
   applies.
3. **Finish auth** (3.3) — `/register` end-to-end.
4. **Add a test suite** (3.4) — there's nothing to regress against right now.
5. **Clean up dead/stub files** (3.2) — mostly safe deletes or quick fills
   once the above are settled.
