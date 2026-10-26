# YOLO vs. YOLOE — Which to Use, and How

`app/engines/vision/vision_engine.py` supports two real detection engines side by
side, selected by config, not code. This guide is the practical "which one, and
how" reference — for the research behind *why* YOLOE exists and its real
accuracy numbers, see `FINE_TUNING_YOLO_GUIDE.md` (Section 2).

## TL;DR

| | **YOLO** (`VISION_MODEL_TYPE=yolo`, default) | **YOLOE** (`VISION_MODEL_TYPE=yoloe`) |
|---|---|---|
| Detects | Fixed 80 COCO classes only | All 80 COCO classes **plus** a ~250-item curated home vocabulary (kitchen, living room, bedroom, bathroom, workspace) — ~297 classes by default, or a custom list via `VISION_CUSTOM_CLASSES` |
| Covers `keys`/`wallet`/`sunglasses`? | No — not COCO classes, no version of YOLO fixes this | Yes, by default |
| Accuracy | 40.9 mAP (YOLO26n, on its known 80 classes) | ~30 mAP (YOLOE-S, zero-shot) — a real, meaningful step down |
| Training/dataset work required | None (pretrained) | None (pretrained, zero-shot) |
| Default weights | `yolo26n.pt` | `yoloe-26n-seg.pt` |

**Use YOLO for anything already in COCO's 80 classes** — more accurate, and that's
what it's built for. **Use YOLOE when you need the broader coverage** (`keys`,
`wallet`, `sunglasses`, or general home items YOLO structurally can't detect) —
accept the accuracy tradeoff in exchange for zero fine-tuning/dataset work.

## Switching from the UI (no restart needed)

The header has a live **YOLO / YOLOE toggle** right next to the camera status
indicator — click either option and the camera feed reconnects with the new
engine within a second or two. No `.env` edit, no container restart.

Mechanically: `CameraFeedDashboard.jsx` tracks the selection and passes it to
`useVisionWS`, which includes it as `?engine=` on the `/ws/vision` WebSocket URL and
reconnects whenever it changes (plain React — `engine` is just a dependency
of the effect that opens the socket). The backend's `get_engine()`
(`app/engines/vision/vision_engine.py`) keeps **both** engines cached once
each has been used, so switching back and forth after the first time is
instant; the very first time you pick an engine in a given backend process,
there's a brief load delay while it constructs — the header shows
"switching…" until the server actually confirms which engine is live (each
detection response carries an `engine` field alongside `detections`), so
what you see reflects what actually served it, not just what you clicked.

## Setting the startup default — env vars

The UI toggle always starts from whatever `VISION_MODEL_TYPE` resolved to at
backend boot. This still matters for the container's initial state, and for
anything outside the running app (scripts, `VisionTracker`, tests) that
doesn't go through `/ws/vision?engine=`. **`docker-compose.yml`'s `backend`
service already wires all four vars through**:

| Var | Applies to | Default | Purpose |
|---|---|---|---|
| `VISION_MODEL_TYPE` | Both | `yolo` | `yolo` or `yoloe` — which engine backs `VisionTracker` |
| `FINE_TUNED_YOLO_MODEL_PATH` | **YOLO only** | unset | Weights file for `YOLOVisionEngine` specifically — use this, not `YOLO_MODEL_PATH`, when pointing at a fine-tuned checkpoint, since both engines can be resident at once and `YOLO_MODEL_PATH` is read by both |
| `YOLO_MODEL_PATH` | Both (legacy) | `yolo26n.pt` (YOLO) / `yoloe-26n-seg.pt` (YOLOE) | Still read by both for back-compat — avoid setting this if you only want to override YOLO's weights, since YOLOE would pick up the same value |
| `VISION_CUSTOM_CLASSES` | YOLOE only | 80 COCO classes + ~250 curated home items (~297 total) | Comma-separated class names passed to `set_classes()` — **replaces** the default entirely, doesn't add to it |

```bash
# .env, then docker compose up -d --no-deps backend (restart only — no rebuild needed,
# these are runtime env vars, not baked into the image)

# Switch to YOLOE, keep default classes
VISION_MODEL_TYPE=yoloe

# Switch to YOLOE with your own class list
VISION_MODEL_TYPE=yoloe
VISION_CUSTOM_CLASSES=keys,wallet,sunglasses

# Back to YOLO (or just remove/comment out VISION_MODEL_TYPE — it's the default)
VISION_MODEL_TYPE=yolo
```

Confirm which engine actually loaded via the backend's logs:
```bash
docker logs -f vision_assist_api
# look for: "YOLOVisionEngine loaded '/opt/weights/yolo26n.pt' (conf>=0.5)."
# or:       "YOLOEVisionEngine loaded '/opt/weights/yoloe-26n-seg.pt' with classes=[...] (conf>=0.5)."
```

`vision_routes.py` also logs which engine class is live the moment a browser
connects to `/ws/vision`, if you'd rather check per-connection than at boot.

## Testing both, side by side

Bring the stack up once (default `VISION_MODEL_TYPE=yolo`), then do
everything else from the browser:

```bash
docker-compose up -d --build backend postgres frontend
docker logs -f vision_assist_api   # wait for boot
```

1. Open the frontend, allow camera access, hold up a phone/keys/wallet with
   the toggle on **YOLO**. `keys`/`wallet` should **not** appear as bounding
   boxes — confirms the class gap.
2. Click **YOLOE** in the header toggle. First click after boot takes a
   moment (constructing the engine); after that it's instant.
3. Same objects, same framing. Expect the same everyday classes (`person`,
   `cell phone`) plus, potentially, `keys`/`wallet`/`sunglasses` — at visibly
   lower confidence scores than YOLO reported for the classes both engines
   know. That's the accuracy tradeoff from the TL;DR table, observed
   directly, not just read about.
4. Click **YOLO** again — instant, since it's still cached from step 1.

## Why not just always use YOLOE?

Since YOLOE can detect *everything* YOLO can plus the custom classes, it's
tempting to make it the default. Don't, for two concrete reasons:

1. **Accuracy regresses on the classes that already work.** `phone`/`backpack`
   detection would get measurably worse (YOLO26n's 40.9 mAP vs. YOLOE's ~30 mAP
   zero-shot) for no benefit, since YOLO already handles those classes well.
2. **It's still not a substitute for fine-tuning if `keys`/`wallet`/`sunglasses`
   need to work *reliably*** — YOLOE is the fast, zero-effort option to prototype
   with; a properly fine-tuned model (see `FINE_TUNING_YOLO_GUIDE.md` — planning
   only, not yet implemented) is the higher-accuracy path if YOLOE's zero-shot
   results prove insufficient for real use, not just a demo.
