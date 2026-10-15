# Fine-tuning YOLO26n for keys/wallet/sunglasses

Implementation of the plan in `docs/FINE_TUNING_YOLO_GUIDE.md`. **Status:
ready to run, not yet run** — no training job has been executed, no
checkpoint produced. This directory has the scripts; running them still
takes a Kaggle session and real GPU time.

## What's here

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Filters LVIS down to key/wallet/sunglasses instances, remaps their class IDs to 80/81/82, folds in a COCO `train2017` sample, writes the merged `lostfound.yaml` |
| `train_kaggle.ipynb` | Kaggle notebook: downloads LVIS+COCO, runs `prepare_dataset.py`, trains with `freeze=10`, sanity-checks the result |
| `evaluate.py` | Three-way comparison harness (base YOLO vs. YOLOE vs. fine-tuned YOLO) against two held-out sets |

## What's being changed, in plain terms

- **Model**: `yolo26n.pt` (2,572,280 parameters total — measured directly, not estimated).
- **Method**: `freeze=10` — not LoRA/QLoRA (see Guide Section 5 for why that was rejected). This freezes every parameter in blocks 0–9 (the Conv/C3k2 backbone stem through SPPF) and leaves the rest trainable:

  | | Tensors | Parameters | % of total |
  |---|---|---|---|
  | Frozen | 99 | 1,115,744 | 43.38% |
  | **Trainable** | 267 | **1,456,536** | **56.62%** |

  Trainable blocks: 10 (`C2PSA` attention), 13/16/17/19/20/22 (neck), 23 (`Detect` head).
- **Classes**: 83 total — the original 80 COCO classes (indices 0–79, untouched) + `keys` (80), `wallet` (81), `sunglasses` (82).
- **Datasets consumed**:
  - **LVIS** (Facebook AI Research, ~160K images, built on COCO's image set) — only the subset of images containing `key` (LVIS class 604), `wallet`/`billfold` (1155), or `sunglasses` (1034); every other LVIS class is dropped from those images' labels, not merged in.
  - **COCO `train2017`** — a random sample (default 750 images, configurable via `--coco-sample-size`) folded in with its original, unmodified 80-class labels, specifically to prevent catastrophic forgetting of the classes the model already knows (Guide Section 6).
  - Fallback if LVIS's filtered counts are too sparse per class: Roboflow Universe community datasets named in Guide Section 3 (not scripted here — a manual download-and-merge step if needed).
- **Not touched**: YOLOE (`yoloe-26n-seg.pt`) — this fine-tune only produces a new YOLO checkpoint; the live UI toggle's YOLOE path is unaffected.

## How to run it

### 1. Prepare the two held-out evaluation sets (do this before training, or in parallel)

Neither is built by any script here — both need a human:

- **Golden set** (~150 images, self-captured): photograph keys/wallet/sunglasses plus `cell phone`/`backpack`/`handbag`/`person` in realistic multi-object scenes, per the per-class instance-count table in Guide Section 7.2. Annotate once in Roboflow's UI (model-assisted auto-labeling pre-boxes the known COCO classes), then export the **same images three times** with different class filters — each model only has output neurons for its own classes, so scoring it against a mismatched label space either crashes or produces a meaningless mAP (Guide Section 7.3):
  - `golden_83class.yaml` — all 83 classes, for the fine-tuned model
  - `golden_80class.yaml` — the 80 COCO classes only, keys/wallet/sunglasses boxes dropped, for base YOLO
  - `golden_3class.yaml` — keys/wallet/sunglasses only, **remapped to class indices 0/1/2** (not 80/81/82) to match what `set_classes()` assigns internally, for YOLOE
- **Forgetting-check set** (~50-100 images, nc=80): a random sample from COCO's own `val2017` — **not** from the golden set, and not from the `train2017` sample folded into training. Export its own `coco_sample.yaml`.

Upload both as Kaggle Datasets so `evaluate.py` can read them from the training notebook's environment (or run `evaluate.py` locally against downloaded copies — it only needs `ultralytics` and `pyyaml`).

### 2. Train on Kaggle (free GPU)

1. Go to kaggle.com → Create → New Notebook.
2. Settings (right sidebar) → Accelerator → **GPU T4 x2** (or P100). No credit card needed, but GPU access requires phone-number verification on your Kaggle account (CPU-only without it).
3. Upload `prepare_dataset.py` as a Kaggle Dataset (or a notebook "Add Data" → "Upload" of just that file) and attach it to the notebook, so cell 2 can call it.
4. Upload `train_kaggle.ipynb` itself (File → Import Notebook), or copy its cells into a fresh Kaggle notebook.
5. Run cells top to bottom:
   - Cell 1 installs `ultralytics==8.4.106` (matches this repo's pin).
   - Cells 2-3 trigger LVIS (~20.7GB) and COCO downloads via one throwaway epoch each — interrupt (Kaggle's stop button) right after each logs its first epoch starting, or let them finish; either way the dataset lands on disk.
   - Cell 4 runs `prepare_dataset.py` to build the merged `lostfound.yaml`.
   - Cell 5 trains: `freeze=10`, `AdamW`, `lr0=0.001`, `patience=15`, 100 epochs (early-stops on plateau). Expect roughly 1-3 hours on a single T4 (Guide Section 8) — well inside Kaggle's session cap (9-12 hours depending on source; check your notebook's own settings) and ~30 GPU-hr/week quota.
   - Cell 6 sanity-checks the result against the merged dataset's own val split (not the real eval — just confirms the new classes learned something).
6. Download `best.pt` from the notebook's Output tab (`/kaggle/working/runs/lostfound_yolo26n/weights/best.pt`).

### 3. Run the real three-way evaluation

Locally, or in another Kaggle/Colab session with `ultralytics` + `pyyaml` installed:

```bash
python evaluate.py \
    --base-weights yolo26n.pt \
    --yoloe-weights yoloe-26n-seg.pt \
    --finetuned-weights best.pt \
    --golden-set-yaml golden_set/golden_83class.yaml \
    --golden-set-base-yaml golden_set/golden_80class.yaml \
    --golden-set-yoloe-yaml golden_set/golden_3class.yaml \
    --coco-sample-yaml coco_sample/coco_sample.yaml \
    --output results.json
```

This produces `results.json` with, per model: mAP@0.5 / mAP@0.5:0.95 (overall and per novel class for the fine-tuned model), the COCO forgetting-check mAP, and latency (mean/p50/p95). Turn it into the comparison table + contact sheet described in Guide Section 7.4.

### 4. Wire the checkpoint into the app

Per Guide Section 9: host `best.pt` somewhere the backend container can fetch it at build time (Hugging Face Hub recommended — don't commit a `.pt` via normal git), add a download `RUN` step to `backend/Dockerfile` mirroring the existing weight pre-downloads, then set `FINE_TUNED_YOLO_MODEL_PATH` (not the shared `YOLO_MODEL_PATH` — see `CLAUDE.md`'s Key Design Patterns) in `docker-compose.yml`'s `backend` service. No application code changes needed beyond that.
