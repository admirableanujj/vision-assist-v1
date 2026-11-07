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
- **Notebook 3** (`03_evaluate_finetune_checks.ipynb`): ✅ complete,
  kernel version 3. See Eval results below for the real (matched
  apples-to-apples) forgetting comparison, built on top of this.
- **Hyperparameter sweep** (in response to the uneven forgetting found
  above): running two variants off the baseline — see Hyperparameter
  sweep section below.

## Dataset & kernel locations

| What | Kaggle ref | Produces |
|---|---|---|
| Notebook 1 kernel (baseline, `--coco-sample-size 750`) | `dceshubh/visionassist-01-dataset-prep-eda` | Output: `lostfound/` (merged 48-class training set), `coco_forgetting_check/` (forgetting-check eval set), `eda_report/` (`class_distribution.csv`/`.png`, `eda_summary.json`) |
| Notebook 1 kernel (sweep, `--coco-sample-size 5000`) | `dceshubh/visionassist-01-dataprep-sample5000` | Same outputs, bigger COCO-only sample |
| Notebook 2 kernel (baseline, `freeze=10`) | `dceshubh/visionassist-02-train` | Output: `runs/lostfound_yolo26n/weights/{best,last}.pt` — also archived locally at `finetuning/runs/run1_baseline_cls0.5_freeze10_sample750/` |
| Notebook 2 kernel (sweep, `freeze=None`, baseline dataset) | `dceshubh/visionassist-02-train-freeze-none` | Same output path, different weights |
| Notebook 2 kernel (sweep, `freeze=10`, sample5000 dataset) | `dceshubh/visionassist-02-train-sample5000` | Same output path, different weights |
| Notebook 3 kernel | `dceshubh/visionassist-03-evaluate-fine-tune-checks` | Forgetting-check + novel-class sanity results |
| **Scripts dataset** (separate from the kernels above) | `dceshubh/vision-assist-finetuning-scripts` | `prepare_dataset.py`, `prepare_coco_forgetting_check.py`, `eda.py` |

## Critical findings (apply to every future run — not just history)

1. **The scripts dataset deploys separately from the notebook.**
   `kaggle kernels push` only updates the notebook file. The actual
   scripts (`prepare_dataset.py` etc.) live in the separate dataset above
   and need their own `kaggle datasets version -p finetuning/kaggle/scripts-dataset`
   after every script change — otherwise the kernel silently runs stale
   code. This sat stale from 2026-11-05 through five supposed notebook-1
   fixes before being caught by notebook 2's first training failure.
2. **`kernel_sources` mount paths are not stable — don't hardcode any
   convention, search instead.** `dataset_sources` (e.g. the scripts
   dataset) reliably mounts at `/kaggle/input/datasets/<owner>/<slug>/`.
   `kernel_sources` is the problem: notebook 2's baseline run mounted its
   single `kernel_sources` entry flat at `/kaggle/input/<slug>/`, but the
   freeze=None sweep variant — identical single `kernel_sources` entry,
   same upstream kernel, no structural difference — mounted the exact
   same thing at `/kaggle/input/notebooks/<owner>/<slug>/` instead, and
   errored fast (`AssertionError`, ~1 min GPU time) on the hardcoded flat
   path. Fixed by replacing every hardcoded `kernel_sources` path with a
   runtime search — `Path("/kaggle/input").rglob(filename)` filtered by
   parent directory name — across `02_train.ipynb` and all its sweep
   variants, and `03_evaluate_finetune_checks.ipynb`. `dataset_sources`
   paths are still hardcoded since they've been reliable every time.
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

## Eval results — notebook 3 (kernel version 3, 2026-11-06)

This is the forgetting-check + novel-class sanity gate, not the real
golden-set benchmark (`finetuning/evaluate.py`, separate, needs the
self-captured photos — see "Open question" below). Completed cleanly,
no errors.

**Novel-class sanity** (fine-tuned `best.pt`, on `lostfound/`'s own val
split): mAP50=0.254, mAP50-95=0.171 — identical to notebook 2's own final
validation (same model, same split, just re-confirmed). Per-class numbers
for the 28 novel classes match training's own breakdown exactly.

**Catastrophic-forgetting check — true apples-to-apples, matched sample.**

Notebook 3's own forgetting-check yaml only used a 75-image sample and
remaps ground truth to this project's compressed 0-19 scheme, which
doesn't line up with base YOLO's standard 80-class output past index 1
(`person`/`bicycle` coincide, everything else diverges) — so base YOLO
can't be scored against that yaml directly, and 75 images isn't the
full picture either way. Built a proper matched comparison instead, same
exact 5,000 COCO val2017 images for both models (downloaded directly —
`images.cocodataset.org/zips/val2017.zip` + the Ultralytics labels zip,
~1GB total, not the full 20GB train+val bundle `coco.yaml`'s own download
block would otherwise pull): base `yolo26n.pt` scored with original
80-class ids, fine-tuned `best.pt` scored with the same 4,952 label files
remapped to the 20 kept classes (reusing `prepare_dataset.py`'s own
`COCO_OLD_TO_NEW` mapping — remapped-image count, 3,471, matches notebook
1's own forgetting-check count exactly, confirming the remap is correct).

| | Base `yolo26n.pt` | Fine-tuned `best.pt` | Relative change |
|---|---|---|---|
| mAP50 (avg over 20 kept classes) | 0.510 | 0.403 | **−20.9%** |
| mAP50-95 (avg over 20 kept classes) | 0.365 | 0.272 | **−25.3%** |

Both scored on the identical 5,000 images, same `imgsz=640`. (An earlier,
mismatched-sample estimate on this same data put the mAP50-95 drop at
~35% — that was comparing a 75-image pre-filtered subset against the
full 4,952; the matched number above supersedes it.)

**The average hides a wide spread — this is the real finding:**

| Class | Train instances | mAP50-95 rel. change |
|---|---|---|
| `person` | 64,070 | **−7.8%** (barely touched) |
| `laptop` | 1,094 | −9.1% |
| `mouse` | 506 | −12.0% |
| `cat` | 297 | −18.1% |
| `backpack` | 2,315 | −18.6% |
| `bicycle` | 1,129 | −21.3% |
| `suitcase` | 1,284 | −39.3% |
| `book` | 3,400 | −44.4% |
| `remote` | 1,059 | −45.7% |
| `scissors` | 180 | **−66.0%** |
| `toothbrush` | 143 | **−66.9%** (worst) |

Full 20-class table (including the 9 classes not shown above — all in
the −18% to −33% range) is in this session's working data, not
duplicated here.

**Reading this:** a real, non-trivial regression, not a collapse —
nothing is near-zero — but the forgetting is very uneven, not uniform.
`scissors`/`toothbrush` (worst-hit) also have by far the fewest training
instances among the 20 kept classes (180/143, vs. `person`'s 64,070),
a real correlation — fewer examples of a kept class surviving the merge
means less to anchor it against forgetting. That doesn't fully explain
it though: `book` (3,400 instances, not sparse) still dropped 44%, and
`remote`/`suitcase` similarly — plausibly visual confusability with
newly-added novel classes that look similar (`book` vs. the new
`notebook`/`passport`; `suitcase` vs. the new `toolbox`/`briefcase`),
but that's a hypothesis, not confirmed — would need a confusion-matrix
look to verify, not asserted here as fact.

**Open question to resolve before golden-set work:** is this acceptable
for gaining 28 new classes, given some kept classes (`scissors`,
`toothbrush`, `book`, `remote`, `suitcase`) lost a lot? Options: accept
it and note the limitation, raise `--coco-sample-size` beyond 750
specifically to give these classes more training weight, or revisit
`freeze=10` → `freeze=None`. Worth deciding before golden-set
photography, not after.

## Hyperparameter sweep (2026-11-06, in progress)

Started in response to the uneven forgetting found above — specifically
to test two hypotheses about *why* `scissors`/`toothbrush`/`book`/
`remote`/`suitcase` regressed so much more than the average. All variants
keep `cls_pw=0.5` fixed (it's free, no reason to toggle it) and
`epochs=100`/`patience=15`/`optimizer=AdamW`/`lr0=0.001` unchanged.

| Run | `freeze` | `--coco-sample-size` | Dataset images | Status |
|---|---|---|---|---|
| **1 — baseline** | 10 | 750 | 18,748 | ✅ done — see Training/Eval results above |
| **2 — bigger COCO sample** | 10 | 5000 | pending (prep running) | running |
| **3 — unfrozen backbone** | None | 750 (same dataset as run 1) | 18,748 | running |

**Run 2's hypothesis:** `merge_and_sample()` draws `coco_sample_size`
COCO-only images **uniformly at random** from the ~64,696-image "has a
kept class, no novel class" pool — not stratified by class. Rare kept
classes (`scissors`: 223 total instances, `toothbrush`: 191) are
under-represented in a small random draw purely because they're rare in
that pool, independent of training quality. Raising the sample ~6.7x
should proportionally raise their representation too. This is a blunt
fix (also pulls in more of the already-dominant classes like `person`),
not the surgical fix (class-stratified sampling) — noted as a possible
follow-up, not implemented.

**Run 3's hypothesis:** `freeze=10` only trains the backbone's attention
block + neck + head (56.62% of params). Letting the full backbone adapt
might recover some of what's lost on kept classes, at the cost of more
compute/epoch and higher overfitting risk on this still-modest dataset.

**Results — filled in once each run's training + forgetting-check complete:**

| Run | mAP50-95 (novel, lostfound val) | mAP50-95 (forgetting-check, matched 5,000-img) | vs. base `yolo26n.pt` (0.365) | Training time |
|---|---|---|---|---|
| 1 — baseline | 0.171 | 0.272 | −25.3% | 4.94hr |
| 2 — sample5000 | pending | pending | pending | pending |
| 3 — freeze=None | pending | pending | pending | pending |

Per-class `scissors`/`toothbrush` mAP50-95 (the worst-hit pair) will be
tracked specifically across runs once available, since that's the direct
test of run 2's hypothesis.

## GPU budget tracking

Kaggle free tier: ~30 GPU-hr/week (T4/P100). Spent: ~4.94hr (notebook 2's
successful baseline training) + ~1 min (its failed first attempt) =
~4.96hr so far. Run 2 (sample5000) adds ~22.6% more images per epoch on
top of the baseline's pace; run 3 (freeze=None) trains more parameters
per step, so expect somewhat longer than baseline for each — actual
times will replace this estimate once both complete. Notebook 3 (eval)
costs zero GPU time regardless of variant (`enable_gpu: false`).
