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
- **Hyperparameter sweep: ✅ complete and conclusive.** Run 2
  (`--coco-sample-size 5000`) improved the forgetting-check on every one
  of the 20 kept classes with only a marginal novel-class cost —
  **recommended as the new best model**, replacing the baseline. Run 3
  (`freeze=None`) made forgetting measurably worse on average and
  catastrophically worse on `toothbrush`/`dog` specifically — discarded.
  See the Catastrophic-forgetting check section below for the full table.
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
| mAP50 (avg, 20 kept classes) | 0.510 | 0.403 | **0.425** | 0.368 |
| mAP50-95 (avg, 20 kept classes) | 0.365 | 0.272 | **0.291** | 0.248 |
| Relative change (mAP50-95) | — | −25.3% | **−20.1%** | −32.0% |

**Result: run 2 (`coco-sample-size=5000`) wins clearly; run 3
(`freeze=None`) is a clear loss.** Full per-class table (mAP50-95,
train-instance counts as in run 1):

| Class | Base | Run 1 | Run 2 | Run 3 | R1 chg | R2 chg | R3 chg |
|---|---|---|---|---|---|---|---|
| `person` | 0.510 | 0.470 | 0.475 | 0.471 | −7.8% | −6.8% | −7.7% |
| `laptop` | 0.598 | 0.543 | 0.546 | 0.504 | −9.1% | −8.6% | −15.6% |
| `mouse` | 0.541 | 0.476 | 0.479 | 0.426 | −12.0% | −11.4% | −21.2% |
| `cat` | 0.717 | 0.587 | 0.609 | 0.484 | −18.1% | −15.0% | −32.5% |
| `backpack` | 0.108 | 0.088 | 0.090 | 0.079 | −18.6% | −16.3% | −26.6% |
| `bicycle` | 0.273 | 0.215 | 0.222 | 0.195 | −21.3% | −18.9% | −28.5% |
| `umbrella` | 0.377 | 0.296 | 0.308 | 0.277 | −21.6% | −18.3% | −26.7% |
| `keyboard` | 0.507 | 0.397 | 0.429 | 0.381 | −21.7% | −15.3% | −24.8% |
| `bottle` | 0.304 | 0.234 | 0.248 | 0.214 | −23.0% | −18.3% | −29.5% |
| `cell phone` | 0.286 | 0.213 | 0.222 | 0.202 | −25.4% | −22.2% | −29.3% |
| `dog` | 0.656 | 0.483 | 0.493 | 0.332 | −26.4% | −24.9% | **−49.5%** |
| `teddy bear` | 0.478 | 0.350 | 0.427 | 0.352 | −26.9% | −10.8% | −26.5% |
| `cup` | 0.355 | 0.258 | 0.284 | 0.234 | −27.5% | −20.0% | −34.1% |
| `handbag` | 0.098 | 0.069 | 0.072 | 0.059 | −29.6% | −27.2% | −39.9% |
| `tie` | 0.296 | 0.198 | 0.232 | 0.217 | −33.1% | −21.6% | −26.7% |
| `suitcase` | 0.388 | 0.236 | 0.263 | 0.204 | −39.3% | −32.1% | −47.5% |
| `book` | 0.098 | 0.054 | 0.063 | 0.056 | −44.4% | −36.0% | −42.8% |
| `remote` | 0.209 | 0.114 | 0.147 | 0.105 | −45.7% | −29.8% | −49.6% |
| `scissors` (180 train inst.) | 0.297 | 0.101 | **0.142** | 0.151 | −66.0% | **−52.1%** | −49.1% |
| `toothbrush` (143 train inst.) | 0.195 | 0.065 | 0.073 | **0.015** | −66.9% | −62.5% | **−92.5%** |

**Takeaways:**
- **Run 2 (bigger COCO sample) improved every single one of the 20 kept
  classes** — not just `scissors`/`toothbrush`, all 20 — confirming its
  hypothesis cleanly. `scissors` specifically: −66.0% → −52.1% (absolute
  mAP50-95 0.101→0.142, +41% relative). `toothbrush`: −66.9% → −62.5%
  (smaller gain — it has the fewest training instances of any kept
  class even after the increase, 325 vs. `scissors`' 320, but started
  from a lower base).
- **Run 3 (unfrozen backbone) made forgetting *worse* on average
  (−32.0% vs. baseline's −25.3%)**, and catastrophically worse on
  `toothbrush` specifically (−92.5%, essentially destroyed — absolute
  mAP50-95 fell to 0.015) and `dog` (−49.5%). Letting the full backbone
  drift away from its COCO-pretrained state amplifies forgetting rather
  than fixing it — this refutes run 3's hypothesis; it was also slower
  to train (86 epochs vs. 55) for a worse outcome on both the novel-class
  metric and this check. Not recommended.
- The `scissors`/`toothbrush`-have-fewest-instances correlation from run
  1 holds up: run 2's gain on `scissors` tracks its +43.5% instance
  increase; `book`/`remote`/`suitcase` (decent instance counts, still
  hit hard in run 1) also improved under run 2, which cuts against pure
  visual-confusability being the main driver there — more COCO-only
  sampling diversity plausibly helps generalization broadly, not just
  via raw rare-class counts. Not fully disentangled; a confusion-matrix
  look would clarify but hasn't been done.

**Recommendation:** adopt run 2 (`finetuning/runs/run2_sample5000_cls0.5_freeze10/`)
over the baseline — it's a strict improvement on every kept class with
only a marginal (0.007) dip in novel-class mAP50-95. Discard run 3.

## GPU budget tracking

Kaggle free tier: ~30 GPU-hr/week (T4/P100). Spent so far this week:
**~19.11hr** — run 1 (4.94hr) + run 2 (6.56hr) + run 3 (7.61hr), plus a
negligible ~1 min from run 3's first attempt that failed fast on the
mount-path bug before any epoch ran. ~11hr of quota remains if further
experiments are needed. Notebook 1 (dataset prep) and notebook 3 (eval)
cost zero GPU time regardless of variant (`enable_gpu: false` for both).
