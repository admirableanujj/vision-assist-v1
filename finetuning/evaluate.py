"""
Three-way comparison: base YOLO26n vs. YOLOE zero-shot vs. fine-tuned YOLO26n.
Implements docs/FINE_TUNING_YOLO_GUIDE.md Section 7.3/7.4.

This project's scheme is 48 classes, NOT the original 80 COCO + 3 novel:
20 of COCO's 80 classes kept (prepare_dataset.KEPT_COCO_CLASS_NAMES, new
indices 0-19) + 28 new home/garage classes (prepare_dataset.NOVEL_CLASSES,
new indices 20-47). See docs/FINE_TUNING_YOLO_GUIDE.md Section 2.

Three models, three DIFFERENT class/index spaces — each needs its OWN
data.yaml, or Ultralytics either crashes or silently produces a
meaningless mAP (see each eval_*() function's docstring for the specific
failure mode):
  --golden-set-yaml       nc=48, this project's own scheme/order exactly
                           (prepare_dataset.LOSTFOUND_CLASS_NAMES) — scores
                           the fine-tuned model, whose output IS this space
  --golden-set-base-yaml  nc=80, ORIGINAL COCO class order (unmodified —
                           do NOT use the 0-19 remap here), only the 20
                           kept-class boxes present — scores base YOLO,
                           whose output is still standard pretrained COCO
  --golden-set-yoloe-yaml nc=28, the 28 novel classes ONLY, in EXACTLY
                           prepare_dataset.NOVEL_CLASSES' order (0-27, not
                           20-47) — matches what set_classes() assigns
                           internally, which starts from 0
  --coco-sample-yaml      nc=48, this project's own scheme (0-19 only ever
                           populated; 20-47 will have zero ground-truth
                           instances, which is correct, not a bug) — a
                           plain COCO val2017 sample needs no manual
                           annotation, so this one IS automated: run
                           prepare_coco_forgetting_check.py to generate it

Golden-set-base-yaml and golden-set-yoloe-yaml still require manual
Roboflow annotation/export (see finetuning/README.md) — only
golden-set-yaml's underlying photos do too; coco-sample-yaml does not.

Usage:
    python evaluate.py \\
        --base-weights yolo26n.pt \\
        --yoloe-weights yoloe-26n-seg.pt \\
        --finetuned-weights runs/lostfound_yolo26n/weights/best.pt \\
        --golden-set-yaml golden_set/golden_48class.yaml \\
        --golden-set-base-yaml golden_set/golden_80class_original_coco.yaml \\
        --golden-set-yoloe-yaml golden_set/golden_28class_novel_only.yaml \\
        --coco-sample-yaml coco_forgetting_check/lostfound.yaml \\
        --output results.json
"""
import argparse
import json
import time
from pathlib import Path

from prepare_dataset import KEPT_COCO_CLASS_NAMES, NOVEL_CLASSES as _NOVEL_CLASS_LVIS_PAIRS

NOVEL_CLASSES = [name for name, _lvis_id in _NOVEL_CLASS_LVIS_PAIRS]
NOVEL_CLASS_INDICES = list(range(len(KEPT_COCO_CLASS_NAMES), len(KEPT_COCO_CLASS_NAMES) + len(NOVEL_CLASSES)))


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
    """Base YOLO26n: unmodified pretrained weights, still outputs standard
    COCO's 80 classes in COCO's own order — golden_set_base_yaml MUST use
    that same nc=80/order (not this project's 0-19 remap), with ground
    truth restricted to the 20 kept classes at their ORIGINAL COCO indices.
    Scoring against this project's own 48-class/order yaml would size
    Ultralytics' confusion matrix at this project's nc and raise IndexError
    the moment a ground-truth box lands at an index past it (or, worse,
    silently misread a kept class as the wrong COCO class if the index
    happened to be in range). The 28 novel classes have no COCO equivalent
    at all and are reported N/A — structural, not a measured 0."""
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
    """YOLOE zero-shot: prompted with the 28 novel classes via
    set_classes(NOVEL_CLASSES), which assigns them output indices 0-27 in
    that exact list order — NOT this project's 20-47. Must be scored
    against an nc=28 yaml using that same 0-27 order, or TP matching never
    lines up with the (differently-indexed) ground truth and reported mAP
    is a meaningless near-zero rather than a real zero-shot measurement."""
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
    """Fine-tuned YOLO26n: full 48-class eval on the golden set (both the
    20 kept-COCO and 28 novel classes) PLUS a separate eval on the COCO
    val2017-derived forgetting-check sample (also this project's own
    48-class scheme — see prepare_coco_forgetting_check.py — its ground
    truth only ever populates indices 0-19, which is correct: a plain COCO
    sample has no novel-class instances). Two different yamls, never reuse
    the training run's own val/ split as either (Section 7.2)."""
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
    parser.add_argument("--golden-set-yaml", required=True, type=Path, help="nc=48, this project's own scheme/order — scores the fine-tuned model")
    parser.add_argument("--golden-set-base-yaml", required=True, type=Path, help="Same images, nc=80, ORIGINAL COCO order (not remapped) — scores base YOLO")
    parser.add_argument("--golden-set-yoloe-yaml", required=True, type=Path, help="Same images, nc=28 (the 28 novel classes only, in NOVEL_CLASSES' exact order) — scores YOLOE")
    parser.add_argument("--coco-sample-yaml", required=True, type=Path, help="nc=48 scheme; generate with prepare_coco_forgetting_check.py")
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
