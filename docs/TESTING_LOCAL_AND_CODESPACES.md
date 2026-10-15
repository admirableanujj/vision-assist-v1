# Testing YOLO / YOLOE — Locally and in GitHub Codespaces

A quick, tested path to actually seeing the vision engine work, in both
environments. Covers two ways to test: a **direct WebSocket check** (fastest,
bypasses login entirely — good for confirming the engine itself works) and
the **full browser flow** (login → camera → live detections).

## Before you start: the DB migration workaround

`precheck_db.py`'s migrations currently fail on a fresh database (see
`GAP_ANALYSIS.md` §3.1 — a bad `VECTOR` column type with no pgvector
extension, among other issues) and the failure is swallowed, so the backend
boots as if nothing went wrong. **Without this workaround, login — and so the
full browser flow — will fail with "relation does not exist."** The direct
WebSocket check below doesn't need this at all, since it never touches auth.

Once `postgres` is up (see setup below), create just enough schema by hand:

```bash
docker exec -i vision_assist_db psql -U postgres -d vision_assist << 'EOF'
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    guid_user VARCHAR(36) UNIQUE,
    username VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE,
    role INT DEFAULT 1,
    status VARCHAR(20) DEFAULT 'active',
    is_active BOOLEAN DEFAULT TRUE,
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_by VARCHAR(50),
    last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_login (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    hashed_password VARCHAR(256) NOT NULL,
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_by VARCHAR(50),
    last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_login_history (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    login_status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_by VARCHAR(50),
    last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (guid_user, username, first_name, last_name, email, role, is_active)
VALUES (gen_random_uuid()::text, 'test_user', 'Test', 'User', 'test@visionassist.local', 1, TRUE)
RETURNING id;
EOF
```

That last `INSERT` prints the new user's `id` — use it below. `auth_routes.py`
checks the password against `salt:sha256(password + salt)`
(`app/core/security.py`'s `verify_password`). Generate one for the password
`test_user`:

```bash
python3 -c "
import hashlib, uuid
salt = uuid.uuid4().hex
pw = 'test_user'
h = hashlib.sha256((pw + salt).encode()).hexdigest()
print(f'{salt}:{h}')
"
```

Then insert it (replace `<id>` with the id printed above, and `<hash>` with
the full `salt:hash` string printed above):

```bash
docker exec vision_assist_db psql -U postgres -d vision_assist \
  -c "INSERT INTO user_login (user_id, hashed_password) VALUES (<id>, '<hash>');"
```

You can now log in with `test_user` / `test_user`. This is a workaround, not
a fix — the real fix belongs in `precheck_db.py` (see `GAP_ANALYSIS.md`).

## Local (Docker)

```bash
git clone https://github.com/admirableanujj/vision-assist-v1.git
cd vision-assist-v1
cp .env.example .env

mkdir -p secrets
printf '%s' 'devpassword123' > secrets/postgres_password.txt
printf '%s' 'unused-for-now' > secrets/qdrant_api_key.txt
```

Edit `.env` and set `POSTGRES_USER=postgres`, `POSTGRES_DB=vision_assist` —
the password itself comes only from `secrets/postgres_password.txt`
(`POSTGRES_PASSWORD_FILE` in `docker-compose.yml`), not from `.env`.

Bring up just what vision testing needs — `--no-deps` skips `backend`'s
`ollama`/`qdrant` dependency check, so you don't need those containers at all:

```bash
docker-compose up -d --build --no-deps backend postgres frontend
docker logs -f vision_assist_api   # wait for "Application startup complete"
```

Apply the DB workaround above, then:

### Option A — direct WebSocket check (fastest, no login needed)

Confirms the engine itself works, independent of the frontend or auth:

```bash
pip install websocket-client   # one-off, local machine
python3 -c "
import websocket, json

ws = websocket.create_connection('ws://localhost:8000/ws/vision')
with open('test_photo.jpg', 'rb') as f:   # any real JPEG with a person/object in it
    ws.send_binary(f.read())
print(json.loads(ws.recv()))
ws.close()
"
```

Expect `{\"status\": \"success\", \"detections\": [...]}`. An empty
`detections` list just means nothing in that photo scored above the 0.5
confidence threshold — try a clearer/closer photo before assuming something's
broken.

### Option B — full browser flow

Open `http://localhost:3000`, log in (`test_user`/`test_user` from the
workaround above), allow camera access. You should see your own feed with
live bounding boxes drawn client-side as `useVisionWS.js` receives detections.

Confirm which engine is actually running:
```bash
docker logs vision_assist_api | grep -i "VisionEngine loaded"
# "YOLOVisionEngine loaded 'yolo26n.pt' (conf>=0.5)." by default
```

### Switching to YOLOE

```bash
echo "VISION_MODEL_TYPE=yoloe" >> .env
docker-compose up -d --no-deps backend   # restart only, no rebuild — env vars are read at import time
docker logs vision_assist_api | grep -i "VisionEngine loaded"
# "YOLOEVisionEngine loaded 'yoloe-26n-seg.pt' with classes=[...] (conf>=0.5)."
```

Re-run Option A or B with the same photo. Expect `keys`/`wallet`/`sunglasses`
to become detectable (if present in frame) and everyday classes both engines
know (e.g. `person`) to score visibly lower than YOLO reported — see
`YOLO_VS_YOLOE_GUIDE.md` for why. Switch back by removing the line from `.env`
and restarting `backend` again.

## GitHub Codespaces

Same steps, with two differences:

1. **Machine size**: the default 2-core/8GB Codespace is tight once
   torch + ultralytics + YOLOE's CLIP dependency are all loaded. Pick at least
   **4-core/16GB** (Codespace creation page, or "Codespaces: Change Machine
   Type" from the VS Code command palette on an existing one).
2. **Opening the app**: Codespaces forwards container ports automatically —
   don't use `localhost:3000`/`localhost:8000`. Open the **Ports** tab in VS
   Code, find ports `3000` and `8000`, and use the forwarded URL it shows
   (something like `https://<codespace-name>-3000.app.github.dev`). The
   frontend's WebSocket hooks already detect `https:` and switch to `wss://`
   automatically (`useVisionWS.js`/`useAudioWS.js`), so this just works —
   no code change needed. If a port shows as Private and the page won't load,
   right-click it in the Ports tab → Port Visibility → Public (fine for a
   personal test Codespace; don't leave it that way on a shared one).

Everything else — the DB workaround, bringing up `backend`/`postgres`/
`frontend`, the direct WebSocket check, switching `VISION_MODEL_TYPE` — is
identical to the local steps above; just run them in the Codespace's terminal
instead of your own.

## Cleanup

```bash
docker-compose down        # stop + remove containers, keep volumes (DB data survives)
docker-compose down -v     # also wipe volumes — full reset, re-apply the DB workaround next time
```
