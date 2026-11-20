# Fine-Tuning Findings: six runs, two rounds, one deployed model

This is the narrative of the YOLO26n lost-and-found fine-tuning: what was
tried, what was measured, what each result changed, and why the deployed model
is the one it is. The design lives in `docs/FINE_TUNING_YOLO_GUIDE.md`, the
raw record in `finetuning/kaggle/RUN_LOG.md`, the per-epoch logs in
`finetuning/runs/*/results.csv`, and an interactive version with charts in the
round-2 report: https://claude.ai/artifact/HGAei6Rwh9KQoDVcC4Uemj

## 1. Summary

- **Problem.** Base YOLO26n only knows COCO's 80 classes, which miss most
  things people lose (`keys`, `wallet`, tools). We fine-tuned it on a curated
  class set built from COCO and LVIS, without making it forget the classes we
  kept.
- **Round 1 (three runs, 48 classes).** Varied the COCO sample size and the
  amount of the network that trains. A bigger COCO sample (5,000 images) helped
  every kept class; unfreezing the whole network made forgetting worse.
- **Review feedback.** `person` is never a lost item and made up 47% of all
  labels; `cls_pw` had only been tried at 0.5; over- and underfitting had not
  been analysed.
- **Round 2 (three runs, 47 classes, no `person`).** Removed `person`, rebuilt
  the data and compared `cls_pw` 0.0, 0.5 and 1.0.
- **Result.** The no-person `cls_pw=0.5` model forgets less than the round-1
  winner (-14.6% vs. -19.6% against base YOLO on the same images), learns the
  28 new classes as well, and is deployed.
- **Fit.** The runs mostly underfit rather than overfit: the limits are the
  small model and the sparse classes, not training too long.

## 2. All six runs

Common settings: `freeze=10` except run 3, AdamW, `lr0=0.001`, `patience=15`,
up to 100 epochs, 640 px, ultralytics 8.4.106, free Kaggle GPU.

| Run | Round | Classes | What varied | `cls_pw` | COCO sample |
|---|---|---|---|---|---|
| 1 | 1 | 48 | baseline | 0.5 | 750 |
| 2 | 1 | 48 | bigger COCO sample | 0.5 | 5,000 |
| 3 | 1 | 48 | `freeze=None` (everything trains) | 0.5 | 750 |
| 4 | 2 | 47 (no person) | `cls_pw` off | 0.0 | 5,000 |
| 5 | 2 | 47 (no person) | `cls_pw` dampened | 0.5 | 5,000 |
| 6 | 2 | 47 (no person) | `cls_pw` full | 1.0 | 5,000 |

Training outcome and loss values at the end of each run (train / validation):

| Run | Epochs | Best epoch | Hours | Best mAP50 | Best mAP50-95 | Box loss | Classification loss | L1 loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 55 | 40 | 4.94 | 0.254 | 0.171 | 1.505 / 1.486 | 0.394 / 0.428 | 0.0070 / 0.0093 |
| 2 | 63 | 48 | 6.56 | 0.250 | 0.164 | 1.486 / 1.458 | 0.383 / 0.421 | 0.0069 / 0.0095 |
| 3 | 86 | 71 | 7.61 | 0.243 | 0.164 | 1.464 / 1.474 | 0.380 / 0.437 | 0.0067 / 0.0092 |
| 4 | 91 | 76 | 7.57 | 0.253 | 0.162 | 1.643 / 1.720 | 1.874 / 2.320 | 0.0063 / 0.0080 |
| 5 | 76 | 61 | 6.59 | 0.255 | 0.165 | 1.695 / 1.697 | 0.664 / 0.746 | 0.0057 / 0.0079 |
| 6 | 100 | 95 | 8.02 | 0.222 | 0.149 | 1.600 / 1.689 | 0.134 / 0.244 | 0.0060 / 0.0078 |

How to read the loss columns:

- Compare **within a run** (train vs. validation) and **within a round**, not
  across all six. Round 2 box loss is higher (about 1.6-1.7 vs. 1.5) because
  `person` boxes were large and easy and are gone, so the remaining objects are
  harder on average.
- Classification loss is not comparable across `cls_pw` values, because
  `cls_pw` rescales that loss itself (1.874 under weight 0.0, 0.664 under 0.5,
  0.134 under 1.0 are the same kind of model, differently weighted). Compare
  those runs by mAP.
- YOLO26 reports box, classification and L1 losses; it has no DFL loss.
- Run 3 early-stopped after 86 epochs of slow progress, about 1.75x the epochs
  of run 1, for a worse result.

## 3. Data: what removing `person` did

Same 22,998 images (18,399 train, 4,599 validation) and the same split seed;
only the labels changed.

| | With person (round 1) | No person (round 2) |
|---|---|---|
| Box instances | 195,425 | 110,529 |
| Largest class | `person`, 91,476 | `shoe`, 10,867 |
| Smallest class | 14 | 14 |
| Imbalance (largest / smallest) | 6,534x | 776x |
| Classes under the 50-instance floor | 5 | the same 5 |

The five sparse classes are `hammer` (46), `toolbox` (44), `tape measure` (41),
`passport` (18) and `game controller` (14). They come from LVIS, so dropping
`person` did not help them; they need new data.

## 4. Over- and underfitting

Overfitting means validation loss rises while training loss keeps falling;
underfitting means both are still falling, or still high, when training stops.
Computed from each run's `results.csv`:

| Run | Final classification loss (train / val) | Gap | Reading |
|---|---|---|---|
| 1 | 0.394 / 0.428 | +9% | underfit, validation still tracking |
| 2 | 0.383 / 0.421 | +10% | underfit, validation still tracking |
| 3 | 0.380 / 0.437 | +15% | underfit, validation still tracking |
| 4 (`cls_pw` 0.0) | 1.874 / 2.320 | +24% | mild classification overfit: validation loss bottoms out at epoch 60 and creeps up while training loss keeps falling |
| 5 (`cls_pw` 0.5) | 0.664 / 0.746 | +12% | underfit, validation still tracking |
| 6 (`cls_pw` 1.0) | 0.134 / 0.244 | +83% | widest gap: up-weighting rare classes lets the model memorise them; also hit the 100-epoch cap with mAP still creeping up |

Conclusions: no run overfits badly; the dominant pattern is underfitting, with
mAP50-95 plateauing around 0.16-0.17. That points at model capacity (YOLO26n is
2.6M parameters) and data quality (sparse classes, many outdoor images) rather
than at too many epochs. Full inverse-frequency weighting (`cls_pw` 1.0) is the
only setting that is both undertrained and overfit on rare classes.

## 5. Catastrophic forgetting, matched

Forgetting is measured on COCO validation images that no run trained on, with
base YOLO26n as the reference (it was never fine-tuned).

**Round 1** (5,000 images, 20 kept classes, mean mAP50-95; from `RUN_LOG.md`):
base 0.365, run 1 0.272 (-25.3%), **run 2 0.291 (-20.1%)**, run 3 0.248
(-32.0%). Run 2 improved all 20 kept classes over run 1; run 3 destroyed
`toothbrush` (-92.5%) and halved `dog`.

**Round 2** (2,051 images, 19 kept classes, mean mAP50-95, every model on the
same images; raw scores in `finetuning/runs/noperson_matched_eval/matched_eval.json`):

| Model | Mean (19 kept classes) | vs. base YOLO26n | Own-val mAP50-95 | Novel classes (28) |
|---|---|---|---|---|
| Base YOLO26n | 0.363 | n/a | n/a | n/a |
| Run 2 (with person, deployed before) | 0.292 | -19.6% | 0.164 | not re-measured |
| Run 4 (`cls_pw` 0.0) | 0.315 | -13.3% | 0.161 | 0.068 |
| **Run 5 (`cls_pw` 0.5, deployed now)** | **0.310** | **-14.6%** | **0.165** | **0.074** |
| Run 6 (`cls_pw` 1.0) | 0.271 | -25.4% | 0.149 | 0.074 |

Harness check: run 2 scores -19.6% on this 2,051-image set, consistent with its
-20.1% on the 5,000-image round-1 set, so the two rounds are comparable.

## 6. Decisions, in order

1. **Bigger COCO sample (run 2 over run 1).** The smaller sample starved rare
   kept classes (`scissors`, `toothbrush`) of images; 5,000 improved all 20
   kept classes at almost no cost to the new classes.
2. **Keep the backbone frozen (run 3 rejected).** Letting everything train made
   forgetting worse (-32.0% vs. -25.3%) and training slower.
3. **Drop `person` (review feedback).** A person is never a lost item and was
   47% of all labels. Result: imbalance 8x lower, kept-class forgetting
   -14.6% instead of -19.6%.
4. **`cls_pw` 0.5 (runs 4-6).** 0.0 ties it on forgetting but is worse on the
   28 new classes and starts to overfit classification; 1.0 is clearly too
   strong. So 0.5 stays.
5. **Deploy run 5.** `backend/weights/lostfound_yolo26n_best.pt` is now the
   round-2 `cls_pw` 0.5 checkpoint. The round-1 run 2 weights remain in
   `finetuning/runs/run2_sample5000_cls0.5_freeze10/`.

## 7. Caveats

- **Not a like-for-like own-validation comparison.** Own-validation mAP is
  computed on different class sets in the two rounds (48 vs. 47 classes), so
  0.164 vs. 0.165 does not by itself show an improvement. The decision rests on
  the matched forgetting check and on the data balance.
- **The deployed model no longer detects people.** The "Fine-tuned" engine now
  ignores them by design; base YOLO still detects them.
- **Run 2's novel-class score was not re-measured** on the round-2 harness, so
  the novel-class column has no run-2 value.
- **Sparse classes remain weak.** Five new classes have under 50 examples and
  the model cannot be expected to learn them.
- **Every run underfits**, and the real golden-set benchmark on self-captured
  photos is still pending.

## 8. Next steps

1. Add data for the five sparse classes (Roboflow Universe or captured photos).
2. Train longer or with a larger model variant, since all runs underfit.
3. Filter or add indoor and garage scenes; many COCO images are outdoors.
4. Run the golden-set benchmark (`finetuning/evaluate.py`) on real photos.
