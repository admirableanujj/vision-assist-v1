# Fine-Tuning YOLO26n: Plan (Not Yet Implemented)

**Status: planning only.** Nothing in this document has been executed — no
dataset downloaded, no training run, no code changed.

## 1. The problem, precisely

Standard MS-COCO — the dataset every public YOLO checkpoint (v5 through the
current YOLO26) is pretrained on — has 80 fixed classes. None of them are
`keys`, `wallet`, or `sunglasses`. No version of YOLO fixes this by being
"better" — they're all trained on the same 80-class label set. `YOLOEVisionEngine`
(see `YOLO_VS_YOLOE_GUIDE.md`) already closes this gap with zero training at
~30% mAP accuracy; this document is the plan for closing it with a real
fine-tuned, higher-accuracy YOLO26n instead, as a third point of comparison.

## 2. Classes

**83 total**: the original 80 COCO classes (indices 0–79, unchanged, standard
order) + 3 appended:

| Index | Class |
|---|---|
| 80 | `keys` |
| 81 | `wallet` |
| 82 | `sunglasses` |

Keeping the original 80 in the label space (rather than training a 3-class-only
model) is deliberate — see Section 6 on preventing catastrophic forgetting.

## 3. Datasets

### Primary: LVIS subset (recommended)

**LVIS** (Large Vocabulary Instance Segmentation, Facebook AI Research) — 1,203
classes over ~160K images, built on COCO's own image set. Confirmed directly
against Ultralytics' current `lvis.yaml`: includes all three needed classes at
**`604: key`, `1155: wallet/billfold`, `1034: sunglasses`**.

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="lvis.yaml", epochs=100, imgsz=640)   # auto-downloads ~20.7GB
```

Don't train on all 1,203 classes — script a subset extraction (images/labels
touching only `key`/`wallet`/`sunglasses`), remap their class IDs to 80/81/82,
and combine with a COCO sample for the original 80 (Section 6).

### Fallback: Roboflow Universe (per-class, if LVIS counts are too sparse)

- Keys: "Keysdetection" by SHEHAB EMAD — 882 images
- Wallet: "wallet" by Mohamed Hussam — 418 images
- Sunglasses: "Sunglasses" by "Data" — 9,997 images, CC BY 4.0

All three confirmed live as of this writing (search-indexed; Roboflow's own
anti-bot protection blocks automated page fetches — open each page manually and
confirm the license before downloading). Sunglasses alone outweighs keys+wallet
combined by ~7–10x — mitigate with undersampling/augmentation, don't train on
the raw counts. Standard checklist before merging any community dataset:
annotation-quality spot check (~50–100 images per source), deduplication,
class-ID remapping to a single consistent scheme, and re-splitting train/val
after merging rather than trusting each source's own split.

## 4. Fine-tuning mechanics

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")          # pretrained COCO weights, not from scratch
model.train(
    data="lostfound.yaml",           # 83-class combined dataset
    epochs=100,
    imgsz=640,
    freeze=10,                       # backbone frozen, neck+head train — see below
    optimizer="AdamW",
    lr0=0.001,
    patience=15,                     # early stopping on val mAP plateau
)
```

`freeze=10` (backbone frozen, only neck + detection head train) is the right
starting point for "small dataset, similar domain to COCO" — Ultralytics' own
guidance. The merged dataset will likely land in the low thousands of images.
Fall back to `freeze=None` if new-class mAP stays weak after ~50 epochs — safer
here since the dataset includes mixed-in COCO examples (Section 6).

## 5. LoRA / QLoRA — researched, not used

**Don't use LoRA/QLoRA for this fine-tune. Use `freeze=10` (above) instead.**

LoRA's entire value proposition is cutting trainable-parameter count and
optimizer memory for models where full fine-tuning doesn't fit in GPU memory —
the problem it solves at 7B+ parameter LLMs. YOLO26n is 2.4M parameters; full
fine-tuning already fits comfortably on a free Kaggle T4/P100. There's no
memory problem to solve. Ultralytics also has **no built-in LoRA/PEFT support**
— two GitHub issues asking for it
([#16983](https://github.com/ultralytics/ultralytics/issues/16983),
[#8728](https://github.com/ultralytics/ultralytics/issues/8728)) are
stale/closed "not planned." `freeze=N` is the only parameter-efficient
mechanism the package ships, and it's already the right tool here.

For context, not adopted here: applying LoRA to convolutional weights is an
active but young research area (LoRA-C, CoLoRA, and especially **YOLO-PEFT**,
which directly benchmarks LoRA vs. full fine-tuning on YOLO11/12, finding LoRA
*can* beat full fine-tuning on accuracy with ~44% less peak memory at the cost
of ~1.7x longer training time — one 2026 paper, not yet a settled result).
Separately, LoRA on a CLIP-style **text encoder** (YOLOE's/YOLO-World's prompt
pathway) is much more mainstream and precedented (YOLO-UniOW does exactly
this) — a legitimate stretch goal for the YOLOE side specifically, not for this
YOLO26n fine-tune.

## 6. Preventing catastrophic forgetting

Fold ~500–1,000 COCO `train2017` images (original labels intact) into the
merged training set. Keep `val2017` untouched by this step — reserved for the
fair-comparison test set in Section 7.

## 7. Evaluation plan — the three-way comparison

The actual goal isn't just "fine-tune a model" — it's a defensible three-way
comparison: **base YOLO26n** (structurally can't detect the 3 classes at all)
vs. **YOLOE zero-shot** (already shipped) vs. **fine-tuned YOLO26n** (this
plan).

### 7.1 Metrics

| Metric | Base YOLO26n | YOLOE zero-shot | Fine-tuned YOLO26n |
|---|---|---|---|
| mAP@0.5 / mAP@0.5:0.95 — keys/wallet/sunglasses | N/A — structural, not a measured 0 | ✅ | ✅ |
| mAP@0.5 / mAP@0.5:0.95 — original 80 COCO classes (forgetting check) | ✅ (reference) | ✅ | ✅ |
| Precision / Recall per class | ✅ | ✅ | ✅ |
| Inference latency (mean / p50 / p95, ms) | ✅ | ✅ | ✅ |
| Qualitative: same test photos, side by side | ✅ | ✅ | ✅ |

Benchmark latency in whatever environment actually serves production inference
for this app (currently: inside the `backend` container, on CPU) — not on the
Kaggle GPU training box.

### 7.2 One fair, held-out test set (no leakage)

- **Novel-class set**: 30–60 fresh, self-captured photos of real
  keys/wallet/sunglasses — leak-free by construction since never in any
  Roboflow/LVIS training split.
- **Forgetting-check set**: a small COCO `val2017` sample (~50–100 images),
  kept separate from the `train2017` sample used in Section 6.
- One `data.yaml` (`nc=83`, same class order as training), reused verbatim for
  the fine-tuned model. Never reuse the fine-tune's own training `val/` split
  as this test set.

### 7.3 Running each variant

Each model needs its own `data.yaml` — they don't share a label space, and
scoring the base model against the 83-class yaml will error or silently
misalign rather than produce a meaningful "0".

```python
# Fine-tuned model — full 83-class evaluation
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="test_set/test_83class.yaml", split="test", imgsz=640)
print(metrics.box.map50, metrics.box.map)
print(metrics.box.maps)   # per-class array; index 80/81/82 = keys/wallet/sunglasses

# Base YOLO26n — scored only on its own 80 classes
base = YOLO("yolo26n.pt")
coco_metrics = base.val(data="test_set/coco80_only.yaml", split="test", imgsz=640)
# Report the keys/wallet/sunglasses row as "N/A — not in COCO label space".

# YOLOE zero-shot — its own 3-class-only data.yaml, matching set_classes()
from ultralytics import YOLOE
yoloe = YOLOE("yoloe-26n-seg.pt")
yoloe.set_classes(["keys", "wallet", "sunglasses"])
yoloe_metrics = yoloe.val(data="test_set/three_class_only.yaml", split="test", imgsz=640)
```

```python
# Identical latency harness for all three
import time

def benchmark(model, frame, conf=0.5, warmup=5, runs=30):
    for _ in range(warmup):
        model.predict(frame, conf=conf, verbose=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": sum(times) / len(times), "p50_ms": times[len(times)//2], "p95_ms": times[int(len(times)*0.95)]}
```

### 7.4 Presenting results

One comparison table (rows = {Base, YOLOE, Fine-tuned}, columns = Section 7.1's
metrics), one qualitative contact sheet (4–6 test photos × three annotated
outputs side by side), and the narrative: base = 0% coverage by construction
(honest baseline, not a weakness of the test); YOLOE = real zero-shot accuracy
at zero training cost; fine-tuned = the accuracy ceiling, at the cost of
dataset curation + GPU time, validated to *not* regress the original 80-class
mAP.

## 8. Free GPU: Kaggle

Kaggle Notebooks free tier: T4/P100, ~30 GPU-hours/week quota, 12-hour session
cap, no credit card required. For a nano model on a low-thousands-of-images
merged dataset, expect roughly 1–3 hours for 50–100 epochs on a single T4.

## 9. Code changes needed in this repo once this is executed

Because `app/engines/vision/vision_engine.py` already resolves `YOLOVisionEngine`'s
weights via the `YOLO_MODEL_PATH` env var (already wired through
`docker-compose.yml`), **no changes to `vision_engine.py` itself are needed** to
serve a fine-tuned checkpoint — `ultralytics.YOLO()` accepts a local file path
directly, and its boxes/confidence flow through the exact same
`_UltralyticsScanMixin.scan_frame()` code already built for the pretrained
model. What's actually needed:

1. Host the fine-tuned `.pt` file somewhere the container can reach (Hugging
   Face Hub recommended — free, versioned, simplest). Don't commit the raw
   `.pt` via normal git.
2. Add a build-time fetch step to `backend/Dockerfile`, mirroring the existing
   `yolo26n.pt`/`yoloe-26n-seg.pt` pre-download `RUN` steps.
3. Point `YOLO_MODEL_PATH` at it in `docker-compose.yml`'s `backend` service —
   zero change to application code.

## Open questions

1. LVIS vs. Roboflow as the primary dataset — fine to start with LVIS and fall
   back if its filtered class counts prove too sparse.
2. How many self-captured test photos (Section 7.2) is realistic to shoot.
3. Checkpoint hosting (Section 9) — Hugging Face Hub assumed, confirm no
   existing team/infra preference overrides that.
