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

## How switching actually works — env vars only (no UI toggle, yet)

**There is currently no runtime UI control to switch engines.** Switching
means changing an environment variable and restarting the `backend`
container. Adding a UI toggle (e.g. a dropdown in `App.jsx` that posts a
"switch engine" message over `/ws/vision`, or a small REST endpoint the frontend
calls before reconnecting) is a reasonable follow-up, not yet built.

`VisionTracker` (resolved once, at module import time, in
`app/engines/vision/vision_engine.py`) picks its engine from three env vars —
**`docker-compose.yml`'s `backend` service already wires all three through**:

| Var | Applies to | Default | Purpose |
|---|---|---|---|
| `VISION_MODEL_TYPE` | Both | `yolo` | `yolo` or `yoloe` — which engine backs `VisionTracker` |
| `YOLO_MODEL_PATH` | Both | `yolo26n.pt` (YOLO) / `yoloe-26n-seg.pt` (YOLOE) | Weights file — same var name for both, interpreted per whichever engine is active |
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
# look for: "YOLOVisionEngine loaded 'yolo26n.pt' (conf>=0.5)."
# or:       "YOLOEVisionEngine loaded 'yoloe-26n-seg.pt' with classes=[...] (conf>=0.5)."
```

`vision_routes.py` also logs which engine class is live the moment a browser
connects to `/ws/vision`, if you'd rather check per-connection than at boot.

## Testing both, side by side

```bash
# 1. Test YOLO first (the default — no .env changes needed)
docker-compose up -d --build backend postgres
docker logs -f vision_assist_api   # wait for boot

# Open the frontend, allow camera access, hold up a phone/keys/wallet.
# keys/wallet should NOT appear as bounding boxes — confirms the class gap.

# 2. Switch to YOLOE
echo "VISION_MODEL_TYPE=yoloe" >> .env
docker-compose up -d --no-deps backend   # restart only, no rebuild
docker logs -f vision_assist_api          # confirm YOLOEVisionEngine loaded with classes=[...]

# Same test, same objects. Expect the same everyday classes (person, cell phone)
# plus, potentially, keys/wallet/sunglasses — at visibly lower confidence scores
# than YOLO reported for the classes both engines know. That's the accuracy
# tradeoff from the TL;DR table, observed directly.

# 3. Switch back
# remove/comment the VISION_MODEL_TYPE line from .env, then:
docker-compose up -d --no-deps backend
```

Each engine switch means a **container restart** (env vars are read once at
module import), not an instant in-app toggle — budget a few seconds per switch.
Revisit this once/if a live UI selector gets built.

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
