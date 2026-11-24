# Qdrant Integration — Zone Camera Stream, Objects Identified & Item Registry

How the frontend's **Camera Stream**, **Zone Inventory → Objects Identified**
and **Zone Inventory → Registered Objects** panels are wired to the Qdrant
vector database (and Postgres), what changed in the code, and what is still
open for discussion.

---

## 1. Requirements covered

| # | Requirement | Where it lives |
|---|---|---|
| 1 | Camera stream → objects identified by YOLO bounding boxes (image + metadata) → persisted in Qdrant `zone_cameras_embeddings` with confidence score | `backend/app/api/vision_routes.py`, `backend/app/engines/ml/zone_camera_store.py` |
| 2 | Zone Inventory → Objects Identified shows the **top 5** objects from the live feed for the selected zone, re-ranked with RRF using confidence | `GET /api/v1/zones/{zone_id}/detected` in `zone_routes.py`, `rrf_top_objects()` |
| 3 | **Register image** button → copy image + metadata from `zone_cameras_embeddings` into `item_embeddings`, return the embedding and store it in Postgres `items.embedding` | `POST /api/v1/items/register` in `zone_routes.py`, `QdrantItemStore.index_item_vectors()` |
| 4 | Clicking a registered item → take its embedding from Postgres `items.embedding`, search Qdrant `item_embeddings`, show results under Objects Identified | `GET /api/v1/items/{item_id}/matches` in `zone_routes.py` |

The YOLO confidence threshold is unchanged at 50% (`DEFAULT_CONFIDENCE_THRESHOLD`
in `vision_engine.py`), so exactly the boxes shown on the stream page are the
ones persisted.

---

## 2. Architecture

```
Camera Stream page (useVisionWS.js)
  │  JPEG frame every ~100 ms
  ▼
WS /ws/vision?engine=..&zone=<zone_id>&token=<JWT>
  │  1. YOLO scan_frame()  ──►  detections sent back to browser (unchanged)
  │  2. per label, best box, max once / ZONE_EMBED_INTERVAL_S (worker thread):
  │       crop → CLIP image embedding → thumbnail
  ▼
Qdrant: zone_cameras_embeddings ───────────────┐
                                               │ GET /zones/{id}/detected
Zone Inventory → Objects Identified ◄──────────┘ (recent sightings → RRF → top 5)
  │
  │  "Register image"  POST /items/register {detection_id}
  ▼
Qdrant: item_embeddings (text + image point)  +  Postgres items.embedding (JSON vector)
  │
  │  click registered item  GET /items/{id}/matches
  ▼
Postgres items.embedding ──► Qdrant item_embeddings search ──► Objects Identified panel
```

Both collections use the same embedding space — **clip-ViT-B-32, 512-d,
COSINE** — so a crop captured on the stream can be registered as an item by
reusing its vector, without running CLIP again.

---

## 3. Qdrant collections

### 3.1 `zone_cameras_embeddings` (new)

One point = one sighting of one object on one camera in one zone.
Point id: random UUID (it is a history of sightings).

| Payload field | Type | Notes |
|---|---|---|
| `zone_id` | int | `zones.id` — indexed (integer) |
| `camera_id` | str | `cameras.id`, or `webcam_zone_<zone_id>` for the browser webcam — indexed (keyword) |
| `label` | str | YOLO class name — indexed (keyword) |
| `description` | str | e.g. `cup seen by camera cam_std_living in zone 1 (91% confidence, yolo engine)` |
| `created_at` | str | ISO-8601 UTC |
| `created_ts` | float | epoch seconds — range index, used for `order_by` (newest first) |
| `confidence_score` | float | YOLO confidence, 0–1 |
| `box` | list | `[x1, y1, x2, y2]` in the 640×480 frame |
| `engine` | str | `yolo` / `finetuned` |
| `owner_id` | int | user from the WebSocket JWT |
| `thumbnail_b64` | str | base64 JPEG of the crop (max 160 px) |

### 3.2 `item_embeddings` (existing sketch, now used)

Each registered item has up to two points with deterministic ids
`uuid5(item_id + modality)`, so re-registering upserts instead of duplicating.

| Payload field | Notes |
|---|---|
| `item_id` | Postgres `items.id` (e.g. `item_f10267a6`) |
| `owner_id` | owning user — every search is filtered by it |
| `item_name` | defaults to the YOLO label |
| `description` | e.g. `cup registered from camera cam_std_living in zone 1` |
| `created_at` | ISO-8601 UTC |
| `modality` | `text` or `image` |
| `text` | `"<item_name>. <description>"` |
| `thumbnail_b64` | image point only — copied from the sighting |
| `zone_id`, `camera_id`, `object_class`, `source_detection_id` | extra traceability back to the sighting |

**No confidence score is stored in `item_embeddings`** (per requirement note c).

---

## 4. Flows in detail

### 4.1 Stream → `zone_cameras_embeddings` (Req. 1)

- `useVisionWS.js` now adds the JWT to the socket URL (`&token=`), because
  browsers cannot send an `Authorization` header on a WebSocket.
- On connect, `vision_routes.py`:
  - decodes the token; no/invalid token → stream works, **nothing is persisted**;
  - checks the zone belongs to the user (`zones.user_id`); foreign zone → not persisted;
  - resolves `camera_id`: the `?camera=` param if it belongs to the zone, else the
    zone's first camera in `cameras`, else `webcam_zone_<zone_id>`.
- Per frame, `_DetectionPersister.offer()`:
  - skips mock/fallback detections (`box: None` — no real crop);
  - keeps only the **highest-confidence box per label**;
  - persists a label at most once every `ZONE_EMBED_INTERVAL_S` (default 5 s);
  - runs crop → CLIP → upsert in a **worker thread** with one job in flight, so
    detection replies to the browser are never delayed;
  - if Qdrant/CLIP fails, logs once and disables persistence for that connection
    (the stream keeps running).

### 4.2 Objects Identified — top 5 with RRF (Req. 2)

`GET /api/v1/zones/{zone_id}/detected?limit=5`

1. Verifies the zone belongs to the user (403 otherwise).
2. Reads the most recent `ZONE_RECENT_POINTS` (default 200) sightings for the
   zone from `zone_cameras_embeddings`, newest first (optionally limited to the
   last `ZONE_WINDOW_SECONDS`).
3. Groups by `label` and ranks each label on three signals, fused with
   weighted Reciprocal Rank Fusion, `score = Σ weight / (60 + rank)`:

   | Signal | Ranking value | Weight |
   |---|---|---|
   | confidence | best `confidence_score` for the label | **1.0** |
   | recency | latest `created_ts` (1-second resolution) | 0.5 |
   | frequency | number of sightings | 0.5 |

   Tied values share a rank, and confidence is the final tie-breaker.
4. Returns the top 5. Each card uses the label's **highest-confidence sighting**
   (best thumbnail) and includes `detection_id`, `confidence`, `camera_id`,
   `sightings`, `rrf_score` and `image` (data URL).
5. If Qdrant is unavailable, falls back to the Postgres `detections` table
   (`"source": "postgres"`, no images).

The page refreshes this list every 9.5 s.

### 4.3 Register image (Req. 3)

`POST /api/v1/items/register` with body `{"detection_id": "<zone_cameras_embeddings point id>"}`
(optional `name`, `description` overrides).

1. Fetches the sighting **with its vector** from `zone_cameras_embeddings` (404 if missing).
2. Verifies the sighting's zone belongs to the user.
3. Inserts the row into Postgres `items`.
4. Writes to `item_embeddings`: the **image point reuses the stored vector and
   thumbnail**; the text point is embedded from `"<label>. <description>"`.
5. Stores the item's primary embedding (the image vector; the text vector for
   text-only registrations) as a JSON array in `items.embedding`, then commits.
6. On any failure: Postgres is rolled back and the Qdrant points are deleted
   (no orphans).

Response:

```json
{
  "status": "success",
  "item_id": "item_f10267a6",
  "item_name": "cup",
  "zone_id": 1,
  "modality": "image",
  "embedding": [0.0123, -0.0456, "... 512 floats"]
}
```

The legacy body (`name`, `zone_id`, optional data-URL `image`) still works for
callers without a `detection_id`.

### 4.4 Registered item → matches (Req. 4)

`GET /api/v1/items/{item_id}/matches?limit=5`

1. Loads `items.embedding` from Postgres for that item and owner (404 if not
   theirs; 409 if no embedding stored).
2. Searches `item_embeddings` with that vector, filtered by `owner_id`.
3. Keeps one result per `item_id` (text and image points are merged); a
   text-point hit borrows the thumbnail from the item's image point.
4. Returns up to 5 matches with `confidence` = cosine similarity, `is_self`
   (the clicked item itself, normally the top hit) and `image` (data URL).

The Objects Identified panel switches to **"Matches: <item>"** mode (live
polling paused) with a **Live feed** button to return.

`GET /api/v1/items/registered` now also returns each item's thumbnail from
`item_embeddings`, so the Registered Objects grid shows the real captured image.

---

## 5. Images: base64 thumbnails → data URLs (note b)

Qdrant stores images as base64 JPEG thumbnails in the payload. ReactJS needs a
URL, so the backend helper `thumbnail_data_url()` (`qdrant_service.py`) returns
`data:image/jpeg;base64,<thumbnail_b64>`. The frontend renders
`item.image_url || item.image`.

**Open for discussion:** moving images to S3/GCS. The hook is already there —
if a payload carries `image_url`, `thumbnail_data_url()` returns it instead of
building a data URL. Moving to object storage would mean: upload the crop at
persist/register time, store the key or URL in the payload as `image_url`, and
optionally stop storing `thumbnail_b64`. No frontend change is needed.

---

## 6. Code changes

### Backend

| File | Change |
|---|---|
| `app/engines/ml/zone_camera_store.py` | **New.** `QdrantZoneCameraStore` (create collection + payload indexes, `add_detection`, `recent_detections`, `get_detection`) and `rrf_top_objects()` |
| `app/engines/ml/qdrant_service.py` | **New.** Lazy shared singletons: CLIP embedder, Qdrant client, `get_item_store()`, `get_zone_camera_store()`; `thumbnail_data_url()` |
| `app/engines/ml/vector_store.py` | Added `index_item_vectors()` (precomputed image vector), `search_by_vector()`, `get_item_points()`, `ensure_payload_indexes()` |
| `app/engines/ml/__init__.py` | Exports `QdrantZoneCameraStore`, `rrf_top_objects` |
| `app/api/vision_routes.py` | `token` / `camera` query params, zone-ownership check, camera resolution, throttled background persistence |
| `app/api/zone_routes.py` | Rewrote `/zones/{id}/detected`, `/items/registered`, `/items/register`; added `/items/{id}/matches`. The new endpoints are sync `def` so CLIP/Qdrant calls run in FastAPI's threadpool |
| `precheck_db.py` | `items.embedding` → `TEXT`; removed `UNIQUE` on `items.user_id` (see §7) |

### Frontend

| File | Change |
|---|---|
| `src/hooks/useVisionWS.js` | Sends the JWT on the vision WebSocket URL |
| `src/pages/ZoneInventoryDashboard.jsx` | Top-5 cards show confidence and sighting count, plus a **Register image** button (sends `detection_id`); registered item click → matches mode with **Live feed** back button; real thumbnails with an icon placeholder when there's no image |

### Config / docs

| File | Change |
|---|---|
| `docker-compose.yml` | Backend gets `QDRANT_URL`, `QDRANT_API_KEY_FILE=/run/secrets/qdrant_api_key`, `ZONE_EMBED_INTERVAL_S` |
| `.env.example` | Documents the Qdrant / zone settings |
| `AGENTS.md` | Qdrant bullet updated (it previously said Qdrant was not wired into any route) |

---

## 7. Database migrations (automatic on backend start)

Added to `precheck_db.py`. Both are idempotent and run on every startup:

```sql
ALTER TABLE items ALTER COLUMN embedding TYPE TEXT;
ALTER TABLE items DROP CONSTRAINT IF EXISTS items_user_id_key;
```

- `embedding VARCHAR(512)` could not hold a JSON-encoded 512-d vector.
- `items.user_id UNIQUE` meant a user's **second** registration failed.

Fresh databases get the corrected `CREATE TABLE` directly.

---

## 8. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant endpoint |
| `QDRANT_API_KEY_FILE` | `/run/secrets/qdrant_api_key` | API key (same Docker secret as the qdrant service); `QDRANT_API_KEY` env also works |
| `QDRANT_COLLECTION` | `item_embeddings` | Item collection name |
| `QDRANT_ZONE_COLLECTION` | `zone_cameras_embeddings` | Zone sightings collection name |
| `ZONE_EMBED_ENABLED` | `true` | Turn stream persistence off entirely |
| `ZONE_EMBED_INTERVAL_S` | `5` | Min seconds between saves of the same label per stream |
| `ZONE_TOP_K` | `5` | Default number of Objects Identified cards |
| `ZONE_RECENT_POINTS` | `200` | Recent sightings fed into RRF |
| `ZONE_WINDOW_SECONDS` | `0` | Only use sightings from the last N seconds (0 = no limit) |
| `EMBEDDING_MODEL_NAME` | `clip-ViT-B-32` | CLIP model (existing) |

---

## 9. Running it

```bash
docker-compose up -d --build   # frontend rebuild + new backend env vars
```

Then: log in → **Camera Stream** → pick a zone, point the camera at objects for
a few seconds → **Zone Inventory** → same zone → top 5 appear → **Register
image** → item appears under Registered Objects → click it to see matches.

---

## 10. Verification done

- **Backend smoke test** with an in-memory Qdrant (`QdrantClient(":memory:")`),
  a stand-in for the CLIP model, SQLite and FastAPI `TestClient`, covering:
  - stream persistence to `zone_cameras_embeddings` with the correct zone/camera,
    best box per label, mock boxes skipped;
  - no persistence without a token or for another user's zone;
  - top-5 RRF ordering (91% `cup` ranks above 66% `keys`), data-URL images, 403 on a foreign zone;
  - registration: `item_embeddings` payload fields present, no confidence stored,
    `items.embedding` equals the returned vector, 404 on an unknown detection;
  - registered list returns thumbnails;
  - matches: self is the top hit, one result per item, 404 for another user.
- **Frontend:** `vite build` succeeds.
- **Not yet done:** an end-to-end run in Docker with the real CLIP model and a
  Qdrant server.

---

## 11. Open items / for discussion

1. **S3/GCS for images** (§5): the `image_url` hook is ready; storage and upload are not built.
2. **CLIP weights are not baked into the Docker image.** `clip-ViT-B-32` downloads
   from Hugging Face on first use; the container needs network access, or the
   model should be pre-downloaded in the Dockerfile like the YOLO weights.
3. **Retention:** `zone_cameras_embeddings` grows without limit (at most ~12
   points/min per label per camera at the default interval). A TTL or cleanup
   job is still needed.
4. **Top 5 is grouped by label:** two different cups in one zone show as one "cup" card.
5. **JWT in the WebSocket query string** is common practice, but it can appear in
   proxy/access logs; a short-lived WebSocket ticket would be stricter.
6. **Placeholder items removed:** the hardcoded Unsplash cards in `/detected` and
   `/items/registered` are gone, since they could not be registered or matched.
