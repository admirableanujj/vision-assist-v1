"""
Three-way comparison: base YOLO26n vs. YOLOE zero-shot vs. fine-tuned YOLO26n.
Implements docs/FINE_TUNING_YOLO_GUIDE.md Section 7.3/7.4.

Needs two held-out sets already prepared as separate directories (see
finetuning/README.md for how to build them):
  --golden-set-yaml   the ~150-image self-captured set (nc=83, same class
                       order as training) — answers "which model actually
                       works for this product"
  --coco-sample-yaml  the untouched COCO val2017 sample (nc=80) — answers
                       "did fine-tuning regress the original classes"

Neither set is produced by this script or by prepare_dataset.py — both
require a human to actually capture/annotate (golden set) or sample from an
already-downloaded COCO val2017 (forgetting-check set).

Usage:
    python evaluate.py \\
        --base-weights yolo26n.pt \\
        --yoloe-weights yoloe-26n-seg.pt \\
        --finetuned-weights runs/lostfound_yolo26n/weights/best.pt \\
        --golden-set-yaml golden_set/golden.yaml \\
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


def eval_base_yolo(weights_path, golden_set_yaml, sample_frame):
    """Base YOLO26n: scored only on the golden set's 80 COCO classes it
    already knows. keys/wallet/sunglasses are reported N/A — structural,
    not a measured 0, since the model has no output neurons for them."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    metrics = model.val(data=golden_set_yaml, split="val", imgsz=640)
    return {
        "model": "base_yolo26n",
        "map50_overall": metrics.box.map50,
        "map50_95_overall": metrics.box.map,
        "novel_classes": {cls: "N/A (not in COCO label space)" for cls in NOVEL_CLASSES},
        "latency": benchmark_latency(model, sample_frame),
    }


def eval_yoloe(weights_path, golden_set_yaml, sample_frame):
    """YOLOE zero-shot: prompted with only the 3 novel classes, per
    FINE_TUNING_YOLO_GUIDE.md Section 7.3 — its own class-only data.yaml,
    not the full 83-class one, since it wasn't prompted with the other 80."""
    from ultralytics import YOLOE

    model = YOLOE(weights_path)
    model.set_classes(NOVEL_CLASSES)
    metrics = model.val(data=golden_set_yaml, split="val", imgsz=640)
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
    parser.add_argument("--golden-set-yaml", required=True, type=Path)
    parser.add_argument("--coco-sample-yaml", required=True, type=Path)
    parser.add_argument("--sample-frame", type=Path, default=None, help="One representative image for latency benchmarking; defaults to the first image in the golden set")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    for yaml_path in (args.golden_set_yaml, args.coco_sample_yaml):
        if not yaml_path.exists():
            raise SystemExit(
                f"{yaml_path} not found. This script doesn't build the held-out sets — "
                "see finetuning/README.md for how to prepare and point at them."
            )

    sample_frame = str(args.sample_frame) if args.sample_frame else _first_image_in_yaml(args.golden_set_yaml)

    results = {
        "base_yolo26n": eval_base_yolo(args.base_weights, str(args.golden_set_yaml), sample_frame),
        "yoloe_zero_shot": eval_yoloe(args.yoloe_weights, str(args.golden_set_yaml), sample_frame),
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
