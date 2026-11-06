# Kaggle Run Log

Distilled state, not a blow-by-blow history. Keeps: current status, dataset/
kernel locations, critical operational findings that would otherwise bite a
future run again, and the actual results (EDA stats, training metrics, eval
results). Superseded numbers get replaced, not appended — this file always
reflects the latest known-good state, not every attempt that got there.

---

## Status (as of 2026-11-06)

- **Notebook 1** (`01_dataset_prep_and_eda.ipynb`): re-running as kernel
  version 14, regenerating output now that the scripts dataset is actually
  fixed (see Finding 1 below). EDA numbers below are from version 13 —
  real, but produced on stale pre-fix scripts (slow due to Finding 1's
  O(n²) bug, though the data itself was correct) — will be replaced once
  v14 completes.
- **Notebook 2** (`02_train.ipynb`): first attempt (kernel version 2)
  failed before any epoch ran (~1 min GPU time) on the stale-scripts bug
  below. Waiting on notebook 1 v14, then retrying.
- **Notebook 3** (`03_evaluate_finetune_checks.ipynb`): not yet run.

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

*From kernel version 13 — correct data, produced on stale pre-fix scripts
(slow, Finding 1). Will be replaced by v14's numbers.*

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

## Training results — notebook 2

Pending. First attempt failed before any epoch (stale-scripts `path:` bug,
now fixed — see Finding 1). No train/val loss or timing yet.

## Eval results — notebook 3

Not yet run.

## GPU budget tracking

Kaggle free tier: ~30 GPU-hr/week (T4/P100). Spent so far: ~1 minute
(notebook 2's failed first attempt — caught before any epoch). Expected
training cost once it actually runs: 1-3hr for 50-100 epochs on a nano
model (`docs/FINE_TUNING_YOLO_GUIDE.md` Section 8).
