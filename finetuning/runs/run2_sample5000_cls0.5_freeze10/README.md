# Run 2 — coco-sample-size=5000 (sweep variant)

Kaggle kernels `dceshubh/visionassist-01-dataprep-sample5000` (dataset) +
`dceshubh/visionassist-02-train-sample5000` (training), 2026-11-07.

## Hyperparameters (vs. run 1 baseline)

| Param | Run 1 (baseline) | Run 2 |
|---|---|---|
| `freeze` | 10 | 10 (unchanged) |
| `cls_pw` | 0.5 | 0.5 (unchanged) |
| `--coco-sample-size` | 750 | **5000** |
| Dataset images | 18,748 | 22,998 |

## Why

Baseline's forgetting-check showed `scissors`/`toothbrush` — the kept
classes with by far the fewest training instances — lost ~66% of their
mAP50-95 relative to base YOLO. `--coco-sample-size` controls how many
COCO-only images (no novel class) `merge_and_sample()` draws **uniformly
at random** from the ~64,696-image candidate pool; raising it pulls in
more of the rare kept classes too, proportionally. Real effect on
training instance counts: `scissors` 223→320 (+43.5%), `toothbrush`
191→325 (+70.2%) — real but more modest than the ~6.7x sample-size
increase alone would suggest, since the LVIS-overlap portion (unchanged)
already contributed part of the original counts.

## Result

Early-stopped at epoch 63 (best: epoch 48), 6.56 hours on a T4.

- Novel-class (lostfound val): mAP50 0.250, mAP50-95 0.164 (vs. run 1's
  0.254 / 0.171 — slightly lower, despite more training data)
- Forgetting-check (matched 5,000-image COCO val2017 comparison): mAP50
  0.425, mAP50-95 0.291 — a −20.1% relative drop vs. base `yolo26n.pt`
  (run 1's was −25.3%). **Improved every one of the 20 kept classes**,
  including the direct test: `scissors` −66.0%→−52.1% (mAP50-95
  0.101→0.142), `toothbrush` −66.9%→−62.5%. Confirms this run's
  hypothesis — **this is the model now deployed in the app**
  (`backend/weights/lostfound_yolo26n_best.pt`).

Full write-up, full per-class table, and the comparison against run 3:
`finetuning/kaggle/RUN_LOG.md`.
