# Run 6: no person, cls_pw=1.0

Round 2 of the fine-tuning (see `docs/FINE_TUNING_FINDINGS.md`). 47 classes
(`LOSTFOUND_DROP_KEPT_CLASSES=person`), `--coco-sample-size 5000`, `freeze=10`,
AdamW, `lr0=0.001`, `patience=15`, up to 100 epochs. Kaggle kernel
`dceshubh/visionassist-02-train-noperson-clspw10`.

Hit the 100-epoch cap; weakest on every measure.

- `results.csv`: per-epoch losses and metrics. `args.yaml`: the trainer arguments.
- `weights/best.pt`: best checkpoint by validation mAP50-95.
