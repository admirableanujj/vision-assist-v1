# Run 1 — baseline

Kaggle kernel `dceshubh/visionassist-02-train`, version 3, 2026-11-06.

## Hyperparameters

| Param | Value |
|---|---|
| `freeze` | 10 |
| `cls_pw` | 0.5 |
| `--coco-sample-size` (dataset prep) | 750 (default) |
| `epochs` (cap) | 100 |
| `patience` | 15 |
| `optimizer` | AdamW |
| `lr0` | 0.001 |

## Result

Early-stopped at epoch 55 (best: epoch 40), 4.939 hours on a T4.

- Novel-class (lostfound val): mAP50 0.254, mAP50-95 0.171
- Forgetting-check (20 kept classes, matched 5,000-image COCO val2017 comparison): mAP50 0.403, mAP50-95 0.272 — base `yolo26n.pt` scores 0.510 / 0.365 on the same images, a −20.9%/−25.3% relative drop
- Worst-hit kept classes: `scissors` (−66.0%), `toothbrush` (−66.9%) — both also have by far the fewest training instances among the 20 kept classes (180/143 vs. `person`'s 64,070)

Full write-up: `finetuning/kaggle/RUN_LOG.md`.
