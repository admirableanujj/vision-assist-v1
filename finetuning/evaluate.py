"""
Three-way comparison: base YOLO26n vs. YOLOE zero-shot vs. fine-tuned YOLO26n.
Implements docs/FINE_TUNING_YOLO_GUIDE.md Section 7.3/7.4.

Needs the SAME golden-set images exported as three class-restricted
data.yaml variants, plus the untouched COCO val2017 sample — none of which
this script or prepare_dataset.py produces; all four require a human to
annotate/export (see finetuning/README.md):
  --golden-set-yaml       full 83-class labels — scores the fine-tuned model
  --golden-set-base-yaml  same images, labels filtered to the 80 COCO
                           classes only (nc=80) — scores base YOLO, which
                           has no output neurons for keys/wallet/sunglasses
                           and would otherwise crash or go out of bounds
                           scoring against an nc=83 dataset
  --golden-set-yoloe-yaml same images, labels filtered to keys/wallet/
                           sunglasses only and remapped to 0/1/2 (nc=3) —
                           matches set_classes()'s own output index order,
                           which is NOT 80/81/82
  --coco-sample-yaml      untouched COCO val2017 sample (nc=80) — answers
                           "did fine-tuning regress the original classes"

Each model is scored only against the data.yaml matching its own class
space — reusing the full 83-class yaml across all three (an earlier version
of this script did this) either raises an IndexError in Ultralytics'
confusion-matrix sizing (base YOLO, nc=80) or silently produces meaningless
near-zero mAP from index-mismatched TP matching (YOLOE, whose set_classes()
output indices are 0/1/2, not 80/81/82).

Usage:
    python evaluate.py \\
        --base-weights yolo26n.pt \\
        --yoloe-weights yoloe-26n-seg.pt \\
        --finetuned-weights runs/lostfound_yolo26n/weights/best.pt \\
        --golden-set-yaml golden_set/golden_83class.yaml \\
        --golden-set-base-yaml golden_set/golden_80class.yaml \\
        --golden-set-yoloe-yaml golden_set/golden_3class.yaml \\
        --coco-sample-yaml coco_sample/coco_sample.yaml \\
        --output results.json
"""
import argparse
import json
import time
from pathlib import Path

NOVEL_CLASSES = ["keys", "wallet", "sunglasses"]
NOVEL_CLASS_INDICES = [80, 81, 82]  # in the 83-class scheme; see prepare_dataset.py


def benchmark_latency(model, frame, conf=0.5, warmup=5, runs=30):
    """Identical latency harness for all three models — Section 7.3."""
    for _ in range(warmup):
        model.predict(frame, conf=conf, verbose=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms": sum(times) / len(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
    }


def eval_base_yolo(weights_path, golden_set_base_yaml, sample_frame):
    """Base YOLO26n: scored on the golden-set images via an nc=80
    class-restricted yaml (the model has no output neurons for
    keys/wallet/sunglasses — scoring it against the full nc=83 yaml sizes
    Ultralytics' confusion matrix at 80 and raises IndexError the moment a
    ground-truth box at class 80/81/82 is encountered). keys/wallet/
    sunglasses are reported N/A — structural, not a measured 0."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    metrics = model.val(data=golden_set_base_yaml, split="val", imgsz=640)
    return {
        "model": "base_yolo26n",
        "map50_overall": metrics.box.map50,
        "map50_95_overall": metrics.box.map,
        "novel_classes": {cls: "N/A (not in COCO label space)" for cls in NOVEL_CLASSES},
        "latency": benchmark_latency(model, sample_frame),
    }


def eval_yoloe(weights_path, golden_set_yoloe_yaml, sample_frame):
    """YOLOE zero-shot: prompted with only the 3 novel classes via
    set_classes(), which assigns them output indices 0/1/2 — NOT 80/81/82.
    Must be scored against an nc=3 yaml whose labels are remapped to match
    those same 0/1/2 indices, or TP matching never lines up with the
    (differently-indexed) ground truth and reported mAP is a meaningless
    near-zero rather than a real zero-shot measurement."""
    from ultralytics import YOLOE

    model = YOLOE(weights_path)
    model.set_classes(NOVEL_CLASSES)
    metrics = model.val(data=golden_set_yoloe_yaml, split="val", imgsz=640)
    return {
        "model": "yoloe_zero_shot",
        "map50_overall": metrics.box.map50,
        "map50_95_overall": metrics.box.map,
        "latency": benchmark_latency(model, sample_frame),
    }


def eval_finetuned(weights_path, golden_set_yaml, coco_sample_yaml, sample_frame):
    """Fine-tuned YOLO26n: full 83-class eval on the golden set (novel-class
    accuracy) PLUS a separate eval on the untouched COCO val2017 sample
    (forgetting check) — two different yamls, never reuse the training
    run's own val/ split as either (Section 7.2)."""
    from ultralytics import YOLO

    model = YOLO(weights_path)

    golden_metrics = model.val(data=golden_set_yaml, split="val", imgsz=640)
    novel_maps = {
        cls: golden_metrics.box.maps[idx]
        for cls, idx in zip(NOVEL_CLASSES, NOVEL_CLASS_INDICES)
    }

    forgetting_metrics = model.val(data=coco_sample_yaml, split="val", imgsz=640)

    return {
        "model": "finetuned_yolo26n",
        "map50_overall_golden_set": golden_metrics.box.map50,
        "map50_95_overall_golden_set": golden_metrics.box.map,
        "novel_classes_map50_95": novel_maps,
        "coco_forgetting_check_map50_95": forgetting_metrics.box.map,
        "latency": benchmark_latency(model, sample_frame),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--yoloe-weights", required=True)
    parser.add_argument("--finetuned-weights", required=True)
    parser.add_argument("--golden-set-yaml", required=True, type=Path, help="nc=83, scores the fine-tuned model")
    parser.add_argument("--golden-set-base-yaml", required=True, type=Path, help="Same images, nc=80, scores base YOLO")
    parser.add_argument("--golden-set-yoloe-yaml", required=True, type=Path, help="Same images, nc=3 (keys/wallet/sunglasses remapped to 0/1/2), scores YOLOE")
    parser.add_argument("--coco-sample-yaml", required=True, type=Path)
    parser.add_argument("--sample-frame", type=Path, default=None, help="One representative image for latency benchmarking; defaults to the first image in the golden set")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    for yaml_path in (args.golden_set_yaml, args.golden_set_base_yaml, args.golden_set_yoloe_yaml, args.coco_sample_yaml):
        if not yaml_path.exists():
            raise SystemExit(
                f"{yaml_path} not found. This script doesn't build the held-out sets — "
                "see finetuning/README.md for how to prepare and point at them."
            )

    sample_frame = str(args.sample_frame) if args.sample_frame else _first_image_in_yaml(args.golden_set_yaml)

    results = {
        "base_yolo26n": eval_base_yolo(args.base_weights, str(args.golden_set_base_yaml), sample_frame),
        "yoloe_zero_shot": eval_yoloe(args.yoloe_weights, str(args.golden_set_yoloe_yaml), sample_frame),
        "finetuned_yolo26n": eval_finetuned(
            args.finetuned_weights, str(args.golden_set_yaml), str(args.coco_sample_yaml), sample_frame
        ),
    }

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")
    print(json.dumps(results, indent=2))


def _first_image_in_yaml(data_yaml: Path) -> str:
    """Resolves a representative frame for the latency benchmark straight
    from the data.yaml's own val/ image directory, so callers don't have to
    pass --sample-frame separately in the common case."""
    import yaml

    spec = yaml.safe_load(data_yaml.read_text())
    base = Path(spec.get("path", data_yaml.parent))
    val_dir = base / spec["val"]
    images = sorted(p for p in val_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No images found under {val_dir} to use for latency benchmarking; pass --sample-frame explicitly.")
    return str(images[0])


if __name__ == "__main__":
    main()
