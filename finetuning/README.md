# Fine-tuning YOLO26n: a curated 48-class home/garage lost & found model

Implementation of the plan in `docs/FINE_TUNING_YOLO_GUIDE.md`. **Status:
ready to run, not yet run** — no training job has been executed, no
checkpoint produced. This directory has the scripts; running them still
takes a Kaggle session, real GPU time, and (for the golden set) real
photography/annotation.

**Not the original 80-COCO-classes-plus-3 scheme.** This fine-tune keeps
only 20 of COCO's 80 classes (small, portable, personally-owned items
plausible in a living room, home, or garage) and adds 28 new classes for
things people actually search for that COCO never covered (keys, wallet,
glasses, watch, garage tools, etc.) — see Guide Section 2 for the full
rationale on what was dropped and why.

## What's here

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Filters LVIS to the 28 novel classes + COCO `train2017` to the 20 kept classes, merges images matching both, writes `lostfound.yaml` (nc=48) |
| `prepare_coco_forgetting_check.py` | Fully automatic — builds the forgetting-check set from COCO `val2017`'s own existing ground truth, no new annotation needed |
| `eda.py` | Per-class instance counts + imbalance flag on the merged dataset — see Guide Section 3.3. Run before training, not after. |
| `01_dataset_prep_and_eda.ipynb` | Kaggle notebook 1/3 — downloads LVIS+COCO, runs both prep scripts above, runs `eda.py` as a review gate. No GPU needed. |
| `02_train.ipynb` | Kaggle notebook 2/3 — trains with `freeze=10`, checkpoint/resume-aware (Guide Section 8.2). GPU required. |
| `03_evaluate_finetune_checks.ipynb` | Kaggle notebook 3/3 — Pass 1 of the eval plan (Guide Section 7): fine-tuning-specific gate (novel-class sanity + catastrophic-forgetting check), before any golden-set work |
| `evaluate.py` | Pass 2 of the eval plan — the real three-way comparison (base YOLO vs. YOLOE vs. fine-tuned YOLO) on the self-captured golden set, run once that exists |

## What's being changed, in plain terms

- **Model**: `yolo26n.pt` (2,572,280 parameters total — measured directly, not estimated).
- **Method**: `freeze=10` — not LoRA/QLoRA (see Guide Section 5 for why that was rejected). This freezes every parameter in blocks 0–9 (the Conv/C3k2 backbone stem through SPPF) and leaves the rest trainable:

  | | Tensors | Parameters | % of total |
  |---|---|---|---|
  | Frozen | 99 | 1,115,744 | 43.38% |
  | **Trainable** | 267 | **1,456,536** | **56.62%** |

  Trainable blocks: 10 (`C2PSA` attention), 13/16/17/19/20/22 (neck), 23 (`Detect` head).
- **Classes**: 48 total.
  - **20 kept from COCO** (new indices 0–19): `person`, `bicycle`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `bottle`, `cup`, `book`, `scissors`, `teddy bear`, `toothbrush`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `cat`, `dog`.
  - **28 new** (indices 20–47): `keys`, `wallet`, `sunglasses`, `glasses`, `watch`, `earphone`, `flashlight`, `glove`, `belt`, `pen`, `notebook`, `calculator`, `stapler`, `battery`, `shoe`, `jacket`, `briefcase`, `passport`, `scarf`, `hat`, `screwdriver`, `wrench`, `pliers`, `hammer`, `toolbox`, `tape measure`, `ladder`, `game controller`. Every one of these is backed by a real, verified LVIS class id — see `prepare_dataset.py`'s `NOVEL_CLASSES` or Guide Section 2's table.
  - **Dropped**: the other 60 COCO classes — wild animals, street/outdoor infrastructure, vehicles, food, and fixed furniture/appliances. None of these are ever really "lost," and training capacity on them was capacity not spent on this product's actual job.
- **Datasets consumed**:
  - **LVIS** (Facebook AI Research, ~160K images, built on COCO's image set) — only images containing at least one of the 28 novel classes; every other LVIS class (of 1,203) is dropped from those images' labels, not merged in.
  - **COCO `train2017`** — a capped random sample (default 750 images, configurable via `--coco-sample-size`) of images containing at least one of the 20 *kept* classes, remapped to 0–19 — the catastrophic-forgetting mitigation (Guide Section 6), now scoped to only the classes actually kept, not all 80.
  - An image matching both sources (LVIS reuses COCO's own JPEGs — a "person holding keys" photo is realistic) gets both sets of labels **merged** into one file, not one source overwriting the other.
  - Fallback if a given novel class's LVIS-filtered count is too sparse: Roboflow Universe, searched live per-class (Guide Section 3.2 — no dataset names are hardcoded here, since they go stale).
- **Not touched**: YOLOE (`yoloe-26n-seg.pt`) — this fine-tune only produces a new YOLO checkpoint; the live UI toggle's YOLOE path is unaffected. The 28 novel classes were chosen to overlap with YOLOE's existing `HOME_ITEM_CLASSES` vocabulary where practical, but the two stay independent.

## How to run it

Three Kaggle notebooks, run in order, each committing its own Output for
the next one to attach as an input (Guide Section 8.1) — not one long
notebook, so a training interruption only costs re-running `02_train.ipynb`,
never re-downloading LVIS (~20.7GB). The golden set (real photography) is
separate manual work that can happen in parallel with any of the three.

### 1. `01_dataset_prep_and_eda.ipynb` — dataset prep + EDA (no GPU needed)

Upload `prepare_dataset.py`, `prepare_coco_forgetting_check.py`, and
`eda.py` as a Kaggle Dataset (or **Add Data → Upload**), attach to the
notebook, run top to bottom:

1. Installs `ultralytics==8.4.106` (matches this repo's pin).
2. Triggers LVIS (~20.7GB) and COCO downloads via one throwaway epoch each
   — interrupt (Kaggle's stop button) right after each logs its first
   epoch starting, or let them finish; either way the dataset lands on
   disk.
3. Runs `prepare_dataset.py` to build the merged `lostfound.yaml` (48
   classes) — see its module docstring for exactly what it merges.
4. Runs `prepare_coco_forgetting_check.py` to build the catastrophic-
   forgetting eval set from COCO `val2017`'s own existing ground truth
   (fully automatic, no annotation needed).
5. Runs `eda.py` (Guide Section 3.3) and displays the resulting chart +
   summary inline. **Review this before continuing** — zero-instance or
   severely under-represented classes should be fixed (raise
   `--coco-sample-size`, or add a Roboflow fallback for a specific sparse
   novel class, Guide Section 3.2) before spending any GPU time.
6. **File → Save Version** — commits `lostfound/`, `coco_forgetting_check/`,
   and `eda_report/` as this notebook's Output.

### 2. `02_train.ipynb` — train on Kaggle GPU (checkpoint/resume-aware)

**Add Data** → attach notebook 1's output, then Settings → Accelerator →
**GPU T4 x2** (or P100) — requires phone-number verification on your
Kaggle account (CPU-only without it). Trains with `freeze=10`, `AdamW`,
`lr0=0.001`, `patience=15`, 100 epochs (early-stops on plateau); expect
roughly 1-3 hours on a single T4 (Guide Section 8) — comfortably inside
Kaggle's session cap and ~30 GPU-hr/week quota, but the notebook doesn't
depend on finishing in one session: if the session gets cut off, interrupt
the cell, **Save Version**, and resume from `last.pt` in a fresh session
(exact steps in the notebook itself and in Guide Section 8.2) rather than
restarting from `yolo26n.pt`. **Save Version** once training finishes, to
hand `runs/lostfound_yolo26n/weights/best.pt` to notebook 3.

### 3. `03_evaluate_finetune_checks.ipynb` — Pass 1: the fine-tuning gate

**Add Data** → attach both notebook 1's and notebook 2's outputs. Runs two
checks, no new photography needed:
- Novel-class sanity (mAP on the training run's own val split, indices
  20-47) — confirms the 28 new classes learned *something*.
- Catastrophic-forgetting check (mAP on the COCO-`val2017`-derived
  forgetting-check set, Guide Section 6) — confirms the 20 kept classes
  didn't regress.

**This has to pass before Step 4** — there's no point capturing a golden
set against a checkpoint that didn't train properly. If it doesn't pass,
go back to `02_train.ipynb` (more epochs, a different freeze depth) or
`01_dataset_prep_and_eda.ipynb` (raise `--coco-sample-size`, or a Roboflow
fallback for a specific sparse class) before continuing.

### 4. Golden set + the real three-way evaluation (Pass 2) — separate, manual, usually local

**Golden set — needs real photography + annotation, no way around it.**
Photograph the 48 classes in realistic multi-object home/garage scenes, per
the tiered instance-count targets in Guide Section 7.2 (roughly 300–400
images total — significantly more than a 3-class plan would need, since
this covers 48 classes). Annotate once in Roboflow's UI (model-assisted
auto-labeling pre-boxes the already-known COCO classes, so manual effort
concentrates on the 28 novel ones), then export the **same images three
times** with different class filters — each model only has output neurons
for its own class space, so scoring it against a mismatched one either
crashes or silently produces a meaningless mAP (Guide Section 7.3):

| Export | nc | Class order | For |
|---|---|---|---|
| `golden_48class.yaml` | 48 | This project's own scheme (same order as `lostfound.yaml`) | Fine-tuned model |
| `golden_80class_original_coco.yaml` | 80 | **Original, unmodified standard COCO order** — do NOT use the 0-19 remap here | Base YOLO (unmodified pretrained weights still output standard COCO) |
| `golden_28class_novel_only.yaml` | 28 | The 28 novel classes only, in the exact order `prepare_dataset.NOVEL_CLASSES` lists them | YOLOE (`set_classes()` assigns indices 0-27 in that same order) |

Getting the base-YOLO export's class order wrong (using the 0-19 remap
instead of COCO's own numbering) is the same category of bug as reusing one
yaml for all three models — it won't error, it'll just silently misalign
indices and produce a meaningless number.

Run `evaluate.py` locally (or in another Kaggle/Colab session) once the
golden set is exported and `best.pt` is downloaded from notebook 2's Output
— it only needs `ultralytics` and `pyyaml`:

```bash
python evaluate.py \
    --base-weights yolo26n.pt \
    --yoloe-weights yoloe-26n-seg.pt \
    --finetuned-weights best.pt \
    --golden-set-yaml golden_set/golden_48class.yaml \
    --golden-set-base-yaml golden_set/golden_80class_original_coco.yaml \
    --golden-set-yoloe-yaml golden_set/golden_28class_novel_only.yaml \
    --coco-sample-yaml coco_forgetting_check/lostfound.yaml \
    --output results.json
```

This produces `results.json` with, per model: mAP@0.5 / mAP@0.5:0.95
(overall, and per novel class for the fine-tuned model), the COCO
forgetting-check mAP (fine-tuned model only), and latency (mean/p50/p95).
Turn it into the comparison table + contact sheet described in Guide
Section 7.4.

### 5. Wire the checkpoint into the app

Per Guide Section 9: host `best.pt` somewhere the backend container can fetch it at build time (Hugging Face Hub recommended — don't commit a `.pt` via normal git), add a download `RUN` step to `backend/Dockerfile` mirroring the existing weight pre-downloads, then set `FINE_TUNED_YOLO_MODEL_PATH` (not the shared `YOLO_MODEL_PATH` — see `CLAUDE.md`'s Key Design Patterns) in `docker-compose.yml`'s `backend` service. No application code changes needed beyond that.
