# Kaggle Run Log

Distilled state, not a blow-by-blow history. Keeps: current status, dataset/
kernel locations, critical operational findings that would otherwise bite a
future run again, and the actual results (EDA stats, training metrics, eval
results). Superseded numbers get replaced, not appended — this file always
reflects the latest known-good state, not every attempt that got there.

---

## Status (as of 2026-11-07)

- **Notebook 1** (dataset prep + EDA): ✅ complete for both the baseline
  (`--coco-sample-size 750`) and the sweep variant (`5000`).
- **Notebook 2** (training): ✅ complete for all 3 runs — baseline,
  `coco-sample-size=5000`, `freeze=None`. All early-stopped before the
  100-epoch cap. Weights archived locally under `finetuning/runs/` (see
  table below) in addition to living in each kernel's Kaggle Output.
- **Notebook 3** (forgetting-check gate): ✅ complete for the baseline.
  The two sweep variants were evaluated with a **local** matched
  comparison instead of a Kaggle notebook-3 run (reused the same harness
  already built for the baseline's apples-to-apples check — no need to
  round-trip through Kaggle for CPU-only inference on weights already
  downloaded).
- **Not yet done:** the real golden-set 3-way benchmark
  (`finetuning/evaluate.py`) — blocked on self-captured, annotated
  photos (`finetuning/README.md` Step 4), not a code or pipeline task.
  App integration is intentionally on hold until that benchmark exists.

## Archived model weights

| Run | `freeze` | `--coco-sample-size` | Local path | Kaggle kernel |
|---|---|---|---|---|
| 1 — baseline | 10 | 750 | `finetuning/runs/run1_baseline_cls0.5_freeze10_sample750/weights/` | `dceshubh/visionassist-02-train` |
| 2 — bigger COCO sample | 10 | 5000 | `finetuning/runs/run2_sample5000_cls0.5_freeze10/weights/` | `dceshubh/visionassist-02-train-sample5000` |
| 3 — unfrozen backbone | None | 750 (same dataset as run 1) | `finetuning/runs/run3_freeze_none_cls0.5_sample750/weights/` | `dceshubh/visionassist-02-train-freeze-none` |

All three trained with `cls_pw=0.5`, `epochs=100` (cap), `patience=15`,
`optimizer=AdamW`, `lr0=0.001`. Each archived run has its own `README.md`
with the full hyperparameter/result detail; this file holds the
cross-run comparison.

## Dataset & kernel locations

| What | Kaggle ref | Produces |
|---|---|---|
| Notebook 1 (baseline) | `dceshubh/visionassist-01-dataset-prep-eda` | `lostfound/` (18,748 images), `coco_forgetting_check/`, `eda_report/` |
| Notebook 1 (sweep) | `dceshubh/visionassist-01-dataprep-sample5000` | Same outputs, `lostfound/` at 22,998 images |
| Notebook 2 (baseline) | `dceshubh/visionassist-02-train` | `runs/lostfound_yolo26n/weights/{best,last}.pt` |
| Notebook 2 (sweep: sample5000) | `dceshubh/visionassist-02-train-sample5000` | Same output path, different weights |
| Notebook 2 (sweep: freeze=None) | `dceshubh/visionassist-02-train-freeze-none` | Same output path, different weights |
| Notebook 3 | `dceshubh/visionassist-03-evaluate-fine-tune-checks` | Forgetting-check + novel-class sanity (baseline only — see Status) |
| **Scripts dataset** (separate from the kernels above) | `dceshubh/vision-assist-finetuning-scripts` | `prepare_dataset.py`, `prepare_coco_forgetting_check.py`, `eda.py` |

## Critical findings (apply to every future run — not just history)

1. **The scripts dataset deploys separately from the notebook.**
   `kaggle kernels push` only updates the notebook file. The actual
   scripts (`prepare_dataset.py` etc.) live in the separate dataset above
   and need their own `kaggle datasets version -p finetuning/kaggle/scripts-dataset`
   after every script change — otherwise the kernel silently runs stale
   code. This sat stale for a full day through five supposed notebook-1
   fixes before being caught by notebook 2's first training failure.
2. **`kernel_sources` mount paths are not stable — don't hardcode any
   convention, search instead.** `dataset_sources` (e.g. the scripts
   dataset) reliably mounts at `/kaggle/input/datasets/<owner>/<slug>/`.
   `kernel_sources` is the problem: the exact same single-`kernel_sources`
   shape mounted flat (`/kaggle/input/<slug>/`) for one kernel push and
   at `/kaggle/input/notebooks/<owner>/<slug>/` for another, with no
   structural difference between the two to predict which applies. Fixed
   by replacing every hardcoded `kernel_sources` path with a runtime
   search — `Path("/kaggle/input").rglob(filename)` filtered by parent
   directory name — across all `02_train*.ipynb` variants and
   `03_evaluate_finetune_checks.ipynb`. `dataset_sources` paths are still
   hardcoded since they've been reliable every time.
3. **`kaggle kernels logs -f` on a long-running job does not reliably
   replay the full history from byte 0.** For short runs it looks like a
   full replay; for a long one (70+ epochs, several MB of log) it
   appears to return a bounded recent window instead — naively enumerating
   "all"-prefixed validation-summary lines in the order they appear in a
   single pull silently mislabels their real epoch numbers (observed:
   epochs that were actually 71-86 got reported as 13-30). Always resolve
   the real epoch number from the nearest preceding `N/100` marker in the
   same pull, never from positional/sequential counting.
4. **Redirect `datasets_dir` off `/kaggle/working`** (20GB quota) to
   `/kaggle/tmp` (large ephemeral scratch). `settings.update()` alone does
   not work — `DATASETS_DIR` is frozen at import time in two separate
   modules; patch `settings.json` on disk and force-reimport `ultralytics.*`.
5. **LVIS re-downloads COCO's own images independently.** Symlink
   `lvis/images → coco/images` to dedupe (download COCO first). Side
   effect: the symlink makes `unzip_file()` think `lvis/` is already
   "fully extracted," skipping the labels zip — force it with
   `exist_ok=True` right after the LVIS trigger.
6. **Trigger dataset download via `check_det_dataset()` directly**, never
   `model.train(epochs=1)` as a download trick — the latter actually
   trains (confirmed: ~17hr projected for one COCO epoch on CPU) since
   nothing interrupts an unattended run.
7. **Generated dataset yamls must omit the `path:` key.** A baked-in
   absolute path breaks the moment the yaml is read from a different
   container (notebook 1 writes under `/kaggle/working/`, notebooks 2/3
   read it back under `/kaggle/input/...`).
8. **`!python script.py` cells need `-u`** (unbuffered) or progress
   prints stay invisible until the process exits, making a slow-but-
   working step indistinguishable from a hang.
9. **`cls_pw` (built-in inverse-class-frequency loss weighting, default
   `0.0`/off) is free** — no extra GPU time, just a loss-term change — so
   enable it on the first real training run rather than retrofitting
   after a baseline. Used `0.5` (dampened, not full `1.0`) throughout —
   see EDA results below for why.
10. **`--coco-sample-size` controls a uniform-random draw, not a full
    retention of COCO examples for the kept classes.** `merge_and_sample()`
    keeps every LVIS-overlap image unconditionally, but only draws
    `coco_sample_size` images **uniformly at random** from the much
    larger "kept-class-only" candidate pool (~64,696 images at baseline).
    Rare kept classes are proportionally under-represented in a small
    random draw purely because they're rare in that pool — nothing to do
    with training quality. Raising the sample size helps proportionally
    but is a blunt instrument (also inflates already-dominant classes);
    true class-stratified sampling would be the surgical fix, not
    implemented here.

## EDA results

| Metric | Baseline (`--coco-sample-size 750`) | Sweep (`5000`) |
|---|---|---|
| Total images | 18,748 (train 14,999 / val 3,749) | 22,998 (train 18,399 / val 4,599) |
| Total box instances | 174,995 | 195,425 |
| Classes w/ 0 instances | 0 | 0 |
| Min / max instances per class | 14 / 79,840 | 14 / 91,476 |
| Imbalance ratio (max/min) | 5702.9x | 6534.0x |
| Classes under 50-instance floor | `hammer`, `toolbox`, `tape measure`, `passport`, `game controller` (unchanged — these are the sparse *novel* classes, not affected by `--coco-sample-size`, which only draws from COCO) | same |

Full per-class breakdowns live in each run's `eda_report/class_distribution.csv`
on Kaggle, not duplicated here. `--coco-sample-size` raising 750→5000 did
measurably help the sparse *kept* classes specifically (not reflected in
the floor table above, which only tracks novel classes): `scissors`
223→320 instances (+43.5%), `toothbrush` 191→325 (+70.2%), `book`
4,300→5,752 (+33.8%) — real but more modest than the ~6.7x sample-size
increase alone would suggest, since the unchanged LVIS-overlap portion
already contributed part of the original counts.

## Training results — own validation split

| Run | Best epoch | Total epochs (early-stopped) | Training time | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1 — baseline | 40 | 55 | 4.94hr | 0.254 | 0.171 |
| 2 — sample5000 | 48 | 63 | 6.56hr | 0.250 | 0.164 |
| 3 — freeze=None | 71 | 86 | 7.61hr | 0.243 | 0.164 |

Neither sweep variant beat the baseline on its own novel-class metric —
run 3 needed ~1.75x the epochs to reach a comparable point, consistent
with unfreezing the backbone slowing convergence as expected. The real
test of both variants is the forgetting-check below, not this number.

**Baseline per-class detail** (unchanged by the sweep, since neither
variant touches the novel classes' training data or weighting):
well-learned — `person` (mAP50 0.754), `cat` (0.719), `laptop` (0.708).
Weak novel classes: `passport`/`game controller` did surprisingly well
despite only 3 training instances each (0.574/0.335 mAP50); `hammer`/
`tape measure`/`toolbox` effectively didn't learn (0.09/0.01/0.001).
Also weak despite being above the 50-instance floor: `battery`,
`pliers`, `screwdriver`, `wrench`, `stapler`, `briefcase` — worth a
Roboflow fallback (guide Section 3.2) for these alongside the 5 flagged
ones if they matter for the product.

## Catastrophic-forgetting check — matched apples-to-apples

Same exact 5,000 COCO val2017 images for every model, never used in any
run's training: base `yolo26n.pt` scored with original 80-class ids,
each fine-tuned model scored with the same 4,952 label files remapped to
the 20 kept classes (`prepare_dataset.py`'s own `COCO_OLD_TO_NEW`
mapping — remapped-image count of 3,471 matches notebook 1's own
forgetting-check count exactly, confirming the remap is correct).
Downloaded COCO val2017 directly (~1GB: `images.cocodataset.org/zips/val2017.zip`
+ the Ultralytics labels zip) rather than through `coco.yaml`'s own
20GB train+val download block. Notebook 3's own forgetting-check (75
pre-filtered images, and can't score base YOLO directly since its yaml
remaps ground truth to the compressed 0-19 scheme) is superseded by this
for anything comparing against base YOLO.

| | Base `yolo26n.pt` | Run 1 (baseline) | Run 2 (sample5000) | Run 3 (freeze=None) |
|---|---|---|---|---|
| mAP50 (avg, 20 kept classes) | 0.510 | 0.403 | *pending* | *pending* |
| mAP50-95 (avg, 20 kept classes) | 0.365 | 0.272 | *pending* | *pending* |
| Relative change (mAP50-95) | — | −25.3% | *pending* | *pending* |

*Runs 2/3's matched comparison is running locally as of this update —
replace once complete.*

**Baseline's regression is real but very uneven, not uniform** — this is
the actual finding that motivated the sweep:

| Class | Train instances (run 1) | mAP50-95 rel. change (run 1) |
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

(9 classes not shown: all in the −18% to −33% range.) `scissors`/
`toothbrush` — worst-hit — also have by far the fewest training
instances among the 20 kept classes, a real, verified correlation. That
doesn't fully explain `book`/`remote`/`suitcase` (decent instance counts,
still hit hard) — a plausible but **unconfirmed** hypothesis is visual
confusability with newly-added novel classes that look similar (`book`
vs. `notebook`/`passport`; `suitcase` vs. `toolbox`/`briefcase`); would
need a confusion-matrix look to verify, not asserted as fact here.

**Once runs 2/3's numbers land**, the direct test is whether `scissors`/
`toothbrush` specifically improved — that's what each sweep variant's
hypothesis predicts, not just the aggregate average.

## GPU budget tracking

Kaggle free tier: ~30 GPU-hr/week (T4/P100). Spent so far this week:
**~19.11hr** — run 1 (4.94hr) + run 2 (6.56hr) + run 3 (7.61hr), plus a
negligible ~1 min from run 3's first attempt that failed fast on the
mount-path bug before any epoch ran. ~11hr of quota remains if further
experiments are needed. Notebook 1 (dataset prep) and notebook 3 (eval)
cost zero GPU time regardless of variant (`enable_gpu: false` for both).
