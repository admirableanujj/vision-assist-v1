# Testing YOLO / YOLOE — Locally and in GitHub Codespaces

A quick, tested path to actually seeing the vision engine work, in both
environments. Covers two ways to test: a **direct WebSocket check** (fastest,
bypasses login entirely — good for confirming the engine itself works) and
the **full browser flow** (login → camera → live detections).

## Quick start (GitHub Codespaces)

Already inside a Codespace opened from this repo (so no `git clone` needed).
Set your machine to **4-core/16GB** first (command palette → "Codespaces:
Change Machine Type") — torch + ultralytics + YOLOE's CLIP dependency is
tight on the 2-core/8GB default.

**Step 1** — env, secrets, build+start, all in one paste. `precheck_db.py`
now creates the full schema and seeds test accounts automatically on first
boot (see "DB seeding" below) — no manual DB workaround needed anymore:

```bash
cp .env.example .env
printf '\nPOSTGRES_USER=postgres\nPOSTGRES_DB=vision_assist\n' >> .env
mkdir -p secrets
printf '%s' 'devpassword123' > secrets/postgres_password.txt
printf '%s' 'unused-for-now' > secrets/qdrant_api_key.txt

docker-compose up -d --build --no-deps backend postgres frontend
docker logs -f vision_assist_api   # wait for "DB migrations completed successfully."

echo ""
echo "Open: https://${CODESPACE_NAME}-3000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
echo "Login: test_user / test_user"
```

Click the printed URL (if the port shows Private and won't load: Ports tab →
right-click port 3000 → Port Visibility → Public), log in. You'll land on
the **Zone Inventory** dashboard — click **Camera Stream** in the header to
reach the live vision feed and allow camera access.

**Step 2 (optional)** — skip login and confirm the vision engine itself
works, independent of the frontend/auth:

```bash
pip install websocket-client
python3 -c "
import websocket, json
ws = websocket.create_connection('ws://localhost:8000/ws/vision')
with open('test_photo.jpg', 'rb') as f:   # any real JPEG with a person/object in it
    ws.send_binary(f.read())
print(json.loads(ws.recv()))
ws.close()
"
```

Expect `{"status": "success", "detections": [...]}`. The rest of this guide
below is the same workflow spelled out step by step (useful for local Docker
setup, understanding what the quick start does, or switching to YOLOE).

## DB seeding — automatic, no workaround needed

`precheck_db.py` runs on every backend boot (`main.py`'s `startup_event`),
creates the full schema (`users`, `zones`, `cameras`, `items`, `detections`,
etc.) if it doesn't exist, and — only on a genuinely empty `users` table —
seeds two accounts:

| Username | Password | Role |
|---|---|---|
| `test_user` | `test_user` | standard user, with a couple of pre-seeded zones/cameras |
| `admin` | `adminpassword123` | admin |

This is idempotent — re-running `docker-compose up` against an existing
Postgres volume skips seeding (`Table datasets contain prior objects.` in
the logs) rather than erroring or duplicating rows. No manual `psql` setup
is needed before logging in; just wait for `docker logs vision_assist_api`
to show `DB migrations completed successfully.` before hitting the login
page.

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

No further DB setup needed — `precheck_db.py` seeds `test_user`/`admin`
automatically on first boot (see "DB seeding" above).

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

Open `http://localhost:3000`, log in (`test_user`/`test_user`, auto-seeded —
see "DB seeding" above). You'll land on the Zone Inventory dashboard; click
**Camera Stream** in the header, then allow camera access. You should see
your own feed with live bounding boxes drawn client-side as
`useVisionWS.js` receives detections.

Confirm which engine is actually running:
```bash
docker logs vision_assist_api | grep -i "VisionEngine loaded"
# "YOLOVisionEngine loaded '/opt/weights/yolo26n.pt' (conf>=0.5)." by default
```

### Switching to YOLOE

```bash
echo "VISION_MODEL_TYPE=yoloe" >> .env
docker-compose up -d --no-deps backend   # restart only, no rebuild — env vars are read at import time
docker logs vision_assist_api | grep -i "VisionEngine loaded"
# "YOLOEVisionEngine loaded '/opt/weights/yoloe-26n-seg.pt' with classes=[...] (conf>=0.5)."
```

Re-run Option A or B with the same photo. Expect `keys`/`wallet`/`sunglasses`
to become detectable (if present in frame) and everyday classes both engines
know (e.g. `person`) to score visibly lower than YOLO reported — see
`YOLO_VS_YOLOE_GUIDE.md` for why. Switch back by removing the line from `.env`
and restarting `backend` again.

## GitHub Codespaces

See the **Quick start** section at the top for the condensed path. Two things
differ from local Docker if you're instead following the step-by-step
sections above in a Codespace: machine size (pick 4-core/16GB — the default
2-core/8GB is tight once torch + ultralytics + YOLOE's CLIP dependency are
all loaded) and how you open the app (use the **Ports** tab's forwarded URL
for port 3000, not `localhost:3000` — the frontend's WebSocket hooks already
detect `https:` and switch to `wss://` automatically, so no code change is
needed). If a port shows as Private and the page won't load, right-click it
in the Ports tab → Port Visibility → Public.

## Cleanup

```bash
docker-compose down        # stop + remove containers, keep volumes (DB data survives)
docker-compose down -v     # also wipe volumes — full reset; precheck_db.py reseeds test_user/admin on next boot
```
