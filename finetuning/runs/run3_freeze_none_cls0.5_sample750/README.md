# Run 3 — freeze=None (sweep variant)

Kaggle kernel `dceshubh/visionassist-02-train-freeze-none`, 2026-11-07.
Trained on the same dataset as run 1 (baseline, `--coco-sample-size 750`).

## Hyperparameters (vs. run 1 baseline)

| Param | Run 1 (baseline) | Run 3 |
|---|---|---|
| `freeze` | 10 | **None** |
| `cls_pw` | 0.5 | 0.5 (unchanged) |
| `--coco-sample-size` | 750 | 750 (unchanged, same dataset) |

## Why

`freeze=10` only trains the backbone's attention block + neck + head
(56.62% of params — see `docs/FINE_TUNING_YOLO_GUIDE.md` Section 4).
Letting the full backbone adapt (`freeze=None`) might recover some of
the forgetting-check regression seen in run 1, at the cost of more
compute per epoch and higher overfitting risk on this still-modest
dataset.

## Result

Early-stopped at epoch 86 (best: epoch 71), 7.61 hours on a T4 — notably
slower to converge than run 1 (more trainable parameters), and needed
~1.75x the epochs to reach a comparable point.

- Novel-class (lostfound val): mAP50 0.243, mAP50-95 0.164 (vs. run 1's
  0.254 / 0.171 — slightly lower, despite training far longer)
- Forgetting-check (matched 5,000-image COCO val2017 comparison): see
  `finetuning/kaggle/RUN_LOG.md`'s Hyperparameter sweep section for the
  full per-class table and direct comparison against run 1.
