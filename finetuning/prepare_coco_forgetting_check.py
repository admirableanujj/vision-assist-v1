"""
Build the catastrophic-forgetting-check set from COCO val2017 — fully
automatic, unlike the golden set (which needs real self-captured photos).

Why this can be automated when the golden set can't: COCO val2017 images
already have ground-truth annotations for all 20 of this project's kept
COCO classes (see prepare_dataset.py's KEPT_COCO_CLASS_NAMES) — there's
nothing to photograph or hand-label, only filtering/remapping/sampling,
the exact same mechanical operation prepare_dataset.py already does for
training's COCO fold-in, just applied to val2017 instead of train2017 (kept
strictly separate from training, per docs/FINE_TUNING_YOLO_GUIDE.md
Section 6).

Output uses this project's own 48-class scheme (nc=48, same `names` order
as prepare_dataset.py's LOSTFOUND_CLASS_NAMES/lostfound.yaml) — NOT COCO's
original 80-class order — because that's the fine-tuned model's actual
output space. A plain COCO val2017 sample naturally has zero ground-truth
instances for the 28 novel classes (20-47); that's expected, not a bug —
the resulting mAP reflects accuracy on only the 20 kept-COCO classes,
which is exactly what "did fine-tuning regress the classes we kept"
means. For scoring the UNMODIFIED base YOLO model instead (nc=80,
original COCO class order, no remapping) — a different, separately
exported yaml is needed; see finetuning/README.md's golden-set export
section, which this script does not produce.

Usage:
    python prepare_coco_forgetting_check.py \\
        --coco-root <datasets_root>/coco \\
        --output-dir ./coco_forgetting_check \\
        --sample-size 75
"""
import argparse
import random
from pathlib import Path

from prepare_dataset import COCO_OLD_TO_NEW, LOSTFOUND_CLASS_NAMES, find_candidates


def write_eval_only_yaml(output_dir: Path):
    """Unlike prepare_dataset.write_data_yaml, this set has no train/ split
    at all — it only ever gets used for model.val(split="val"). Points
    `train` at the same images/val/ dir rather than a nonexistent folder,
    since Ultralytics' yaml loader expects both keys to resolve to a real
    path even when `train` is never actually read for a pure eval run."""
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/val\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(LOSTFOUND_CLASS_NAMES)}\n"
        f"names: {LOSTFOUND_CLASS_NAMES!r}\n"
    )
    (output_dir / "lostfound.yaml").write_text(yaml_content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-root", required=True, type=Path, help="Path to Ultralytics' downloaded coco/ dataset root")
    parser.add_argument("--output-dir", default=Path("./coco_forgetting_check"), type=Path)
    parser.add_argument("--sample-size", type=int, default=75, help="Guide Section 7.2 recommends ~50-100 images")
    parser.add_argument("--seed", type=int, default=43)  # different default seed than prepare_dataset.py's training split, deliberately uncorrelated
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_images, out_labels = args.output_dir / "images", args.output_dir / "labels"
    (out_images / "val").mkdir(parents=True, exist_ok=True)
    (out_labels / "val").mkdir(parents=True, exist_ok=True)

    print("Scanning COCO val2017 labels for the 20 kept classes...")
    candidates = find_candidates(args.coco_root / "labels" / "val2017", args.coco_root / "images" / "val2017", COCO_OLD_TO_NEW)
    print(f"Found {len(candidates)} val2017 images containing at least one kept class.")
    if not candidates:
        raise SystemExit(f"No val2017 images matched any kept class. Check --coco-root points at Ultralytics' actual downloaded dataset.")

    sampled_keys = rng.sample(list(candidates), k=min(args.sample_size, len(candidates)))
    sampled = {k: candidates[k] for k in sampled_keys}

    # Entire sample goes in "val" — this set is only ever used for eval
    # (model.val(split="val")), never for training, so there's no
    # train/val split to make here.
    for _stem, entry in sampled.items():
        image_path = entry["image_path"]
        (out_images / "val" / image_path.name).write_bytes(image_path.read_bytes())
        (out_labels / "val" / f"{image_path.stem}.txt").write_text("\n".join(entry["lines"]) + "\n")

    write_eval_only_yaml(args.output_dir)
    print(f"\nDone. {len(sampled)} images. Wrote {args.output_dir / 'lostfound.yaml'} "
          f"(nc=48 — pass this as --coco-sample-yaml to evaluate.py).")


if __name__ == "__main__":
    main()
