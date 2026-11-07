# Kaggle Run Log

Distilled state, not a blow-by-blow history. Keeps: current status, dataset/
kernel locations, critical operational findings that would otherwise bite a
future run again, and the actual results (EDA stats, training metrics, eval
results). Superseded numbers get replaced, not appended — this file always
reflects the latest known-good state, not every attempt that got there.

---

## Status (as of 2026-11-06)

- **Notebook 1** (`01_dataset_prep_and_eda.ipynb`): ✅ complete, kernel
  version 14 — regenerated with the real fix (Finding 1) actually deployed.
  Scan steps that took 34.5min/157min on stale scripts now complete in
  seconds; data itself unchanged (confirms it was always correct, only
  speed was broken).
- **Notebook 2** (`02_train.ipynb`): ✅ complete, kernel version 3 —
  trained successfully after the scripts-dataset fix. Early-stopped at
  epoch 55 (best: epoch 40). See Training results below.
- **Notebook 3** (`03_evaluate_finetune_checks.ipynb`): running (kernel
  version 3, CPU-only, no GPU quota spent — `enable_gpu: false` confirmed
  directly in its `kernel-metadata.json`).

## Dataset & kernel locations

| What | Kaggle ref | Produces |
|---|---|---|
| Notebook 1 kernel | `dceshubh/visionassist-01-dataset-prep-eda` | Output: `lostfound/` (merged 48-class training set), `coco_forgetting_check/` (forgetting-check eval set), `eda_report/` (`class_distribution.csv`/`.png`, `eda_summary.json`) |
| Notebook 2 kernel | `dceshubh/visionassist-02-train` | Output: `runs/lostfound_yolo26n/weights/{best,last}.pt` |
| Notebook 3 kernel | `dceshubh/visionassist-03-evaluate-fine-tune-checks` | not yet run |
| **Scripts dataset** (separate from the kernels above) | `dceshubh/vision-assist-finetuning-scripts` | `prepare_dataset.py`, `prepare_coco_forgetting_check.py`, `eda.py` |

## Critical findings (apply to every future run — not just history)

1. **The scripts dataset deploys separately from the notebook.**
   `kaggle kernels push` only updates the notebook file. The actual
   scripts (`prepare_dataset.py` etc.) live in the separate dataset above
   and need their own `kaggle datasets version -p finetuning/kaggle/scripts-dataset`
   after every script change — otherwise the kernel silently runs stale
   code. This sat stale from 2026-11-05 through five supposed notebook-1
   fixes before being caught by notebook 2's first training failure.
2. **Mount paths differ by attachment type:**
   - `dataset_sources` (e.g. the scripts dataset) → `/kaggle/input/datasets/<owner>/<slug>/`
   - `kernel_sources` (another kernel's output, e.g. notebook 1's output
     attached to notebook 2) → flat `/kaggle/input/<slug>/`
3. **Redirect `datasets_dir` off `/kaggle/working`** (20GB quota) to
   `/kaggle/tmp` (large ephemeral scratch). `settings.update()` alone does
   not work — `DATASETS_DIR` is frozen at import time in two separate
   modules; patch `settings.json` on disk and force-reimport `ultralytics.*`.
4. **LVIS re-downloads COCO's own images independently.** Symlink
   `lvis/images → coco/images` to dedupe (download COCO first). Side
   effect: the symlink makes `unzip_file()` think `lvis/` is already
   "fully extracted," skipping the labels zip — force it with
   `exist_ok=True` right after the LVIS trigger.
5. **Trigger dataset download via `check_det_dataset()` directly**, never
   `model.train(epochs=1)` as a download trick — the latter actually
   trains (confirmed: ~17hr projected for one COCO epoch on CPU) since
   nothing interrupts an unattended run.
6. **Generated dataset yamls must omit the `path:` key.** A baked-in
   absolute path breaks the moment the yaml is read from a different
   container (notebook 1 writes under `/kaggle/working/`, notebooks 2/3
   read it back under `/kaggle/input/...`).
7. **`!python script.py` cells need `-u`** (unbuffered) or progress
   prints stay invisible until the process exits, making a slow-but-
   working step indistinguishable from a hang.
8. **`cls_pw` (built-in inverse-class-frequency loss weighting, default
   `0.0`/off) is free** — no extra GPU time, just a loss-term change — so
   enable it on the first real training run rather than retrofitting
   after a baseline. Currently set to `0.5` (dampened, not full `1.0`) —
   see EDA results below for why.

## EDA results — notebook 1 output (`lostfound/`)

*Confirmed identical on kernel version 14 (real fix deployed) as on v13
(stale scripts) — the underlying data was always correct; only the scan
speed was affected by Finding 1's bug.*

| Metric | Value |
|---|---|
| Total images | 18,748 (train 14,999 / val 3,749) |
| Total box instances | 174,995 |
| Classes | 48 (zero with 0 instances) |
| Min / max instances per class | 14 / 79,840 |
| Imbalance ratio (max/min) | 5702.9x |
| Classes under 50-instance floor | `hammer`, `toolbox`, `tape measure`, `passport`, `game controller` |

Full per-class breakdown lives in `eda_report/class_distribution.csv` on
Kaggle, not duplicated here.

## Training results — notebook 2 (kernel version 3, 2026-11-06)

**Early-stopped at epoch 55** (`patience=15`, no improvement after epoch
40) — **4.939 hours** on a single T4. `cls_pw=0.5` confirmed active
throughout (verified in the saved checkpoint's own `train_args`).

Overall (`best.pt`, on `lostfound/`'s own val split — 3,749 images, 34,773
instances):

| Metric | Value |
|---|---|
| Precision | 0.379 |
| Recall | 0.265 |
| mAP50 | 0.254 |
| mAP50-95 | 0.171 |

**Well-learned classes** (common, high instance count): `person` (mAP50
0.754), `cat` (0.719), `laptop` (0.708), `teddy bear` (0.627).

**The 5 EDA-flagged sparse classes — mixed, mostly as predicted:**

| Class | Instances | mAP50 | mAP50-95 |
|---|---|---|---|
| `passport` | 3 | 0.574 | 0.527 |
| `game controller` | 3 | 0.335 | 0.335 |
| `hammer` | 13 | 0.091 | 0.044 |
| `tape measure` | 17 | 0.011 | 0.005 |
| `toolbox` | 6 | 0.001 | 0.0001 |

`passport`/`game controller` did surprisingly well despite tiny counts
(likely simple, visually distinct appearance) — `hammer`/`tape measure`/
`toolbox` effectively didn't learn. **Other weak classes not caught by
the 50-instance floor:** `battery` (32 inst., mAP50 0.002), `pliers` (18,
0.040), `screwdriver` (18, 0.060), `wrench` (15, 0.066), `stapler` (15,
0.090), `briefcase` (16, 0.034) — worth tightening the floor or adding a
Roboflow fallback (Section 3.2) for these alongside the originally
flagged 5 before relying on this model for those classes specifically.

Full 48-class breakdown + training curves: `runs/lostfound_yolo26n/` in
notebook 2's committed Output (`results.csv`, `results.png`,
`weights/{best,last}.pt`, both 5.4MB).

## Eval results — notebook 3

Running (kernel version 3) — this is the forgetting-check + novel-class
sanity gate, not the real golden-set benchmark (`finetuning/evaluate.py`,
separate, needs self-captured photos). Results pending.

## GPU budget tracking

Kaggle free tier: ~30 GPU-hr/week (T4/P100). Spent: ~4.94hr (notebook 2's
successful training run) + ~1 min (its failed first attempt). Notebook 3
costs zero GPU time (`enable_gpu: false`, CPU inference only).
