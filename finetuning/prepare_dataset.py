"""
Build the 83-class lostfound/ training dataset: LVIS's key/wallet/sunglasses
instances (remapped to classes 80/81/82) + a COCO train2017 sample (original
0-79 labels, for catastrophic-forgetting prevention — see
docs/FINE_TUNING_YOLO_GUIDE.md Section 6).

Run on Kaggle, inside the training notebook, AFTER Ultralytics has already
populated its standard LVIS dataset layout on disk — this script only
filters/remaps/merges, it doesn't download LVIS itself. Trigger that first:

    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")
    model.train(data="lvis.yaml", epochs=1)  # downloads + converts to YOLO format, then stop

(Letting one epoch start is the simplest reliable way to force Ultralytics'
download/convert step to run; interrupt after it logs the first epoch, or
just let it run — the throwaway checkpoint is harmless.) That leaves
Ultralytics' standard dataset root (wherever its settings.yaml points — on
Kaggle this defaults under /kaggle/working) populated with:

    <datasets_root>/lvis/images/{train2017,val2017}/*.jpg
    <datasets_root>/lvis/labels/{train2017,val2017}/*.txt   # YOLO format, LVIS's 1203-class IDs
    <datasets_root>/coco/images/{train2017,val2017}/*.jpg
    <datasets_root>/coco/labels/{train2017,val2017}/*.txt   # YOLO format, COCO's 80-class IDs

Usage:
    python prepare_dataset.py \\
        --lvis-root <datasets_root>/lvis \\
        --coco-root <datasets_root>/coco \\
        --output-dir ./lostfound \\
        --coco-sample-size 750

Produces:
    ./lostfound/images/{train,val}/*.jpg
    ./lostfound/labels/{train,val}/*.txt   # remapped to this project's 83-class scheme
    ./lostfound/lostfound.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

# LVIS class ID -> this project's class ID (indices 80-82, see
# docs/FINE_TUNING_YOLO_GUIDE.md Section 2). Confirmed against Ultralytics'
# own lvis.yaml: 604=key, 1155=wallet/billfold, 1034=sunglasses.
LVIS_TO_LOSTFOUND = {604: 80, 1155: 81, 1034: 82}

# The 80 standard COCO classes, in the exact order every COCO-pretrained YOLO
# checkpoint uses — must match vision_engine.py's COCO_80_CLASSES order
# exactly, since this fixes classes 0-79 of the 83-class scheme.
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
LOSTFOUND_CLASS_NAMES = COCO_80_CLASSES + ["keys", "wallet", "sunglasses"]


def filter_and_remap_lvis_label(label_path: Path) -> list[str]:
    """
    Reads one LVIS YOLO-format label file and returns only the lines whose
    class is one of our 3 target classes, remapped to 80/81/82.

    Critical correctness point: LVIS has 1203 classes. An image containing a
    key ALSO likely has boxes for dozens of other LVIS classes we don't want
    (LVIS's long-tail vocabulary has no relationship to our 83-class scheme's
    indices) — those lines must be dropped entirely, not just left alone,
    or they'd silently corrupt training as wrong-class boxes under our much
    smaller label space (e.g. LVIS class 42 surviving untouched would be
    misread as *our* class 42, an unrelated COCO class).
    """
    kept = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        lvis_class_id = int(parts[0])
        if lvis_class_id in LVIS_TO_LOSTFOUND:
            new_class_id = LVIS_TO_LOSTFOUND[lvis_class_id]
            kept.append(f"{new_class_id} {' '.join(parts[1:])}")
    return kept


def find_lvis_candidates(lvis_labels_dir: Path) -> dict[Path, list[str]]:
    """Scans every LVIS label file; returns {label_path: [kept remapped lines]}
    only for images that contain at least one of our 3 target classes."""
    candidates = {}
    for label_path in sorted(lvis_labels_dir.rglob("*.txt")):
        kept_lines = filter_and_remap_lvis_label(label_path)
        if kept_lines:
            candidates[label_path] = kept_lines
    return candidates


def copy_lvis_subset(lvis_root: Path, candidates: dict, out_images: Path, out_labels: Path, val_fraction: float, rng: random.Random):
    """Copies each candidate's image + its filtered/remapped label into
    out_images/{train,val} and out_labels/{train,val}, assigning the split
    fresh here rather than trusting LVIS's own split (see
    FINE_TUNING_YOLO_GUIDE.md Section 3's "re-split after merging" guidance)."""
    items = list(candidates.items())
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_fraction))
    splits = {"val": items[:n_val], "train": items[n_val:]}

    counts = {"train": 0, "val": 0}
    for split, split_items in splits.items():
        for label_path, lines in split_items:
            # LVIS label paths mirror image paths one-for-one, just under
            # labels/ instead of images/, with the same stem.
            rel = label_path.relative_to(lvis_root / "labels")
            image_candidates = list((lvis_root / "images" / rel.parent).glob(f"{label_path.stem}.*"))
            image_candidates = [p for p in image_candidates if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not image_candidates:
                print(f"[WARN] No image found for label {label_path}, skipping")
                continue
            image_path = image_candidates[0]

            dest_image = out_images / split / image_path.name
            dest_label = out_labels / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dest_image)
            dest_label.write_text("\n".join(lines) + "\n")
            counts[split] += 1
    return counts


def sample_coco_fold_in(coco_root: Path, sample_size: int, out_images: Path, out_labels: Path, val_fraction: float, rng: random.Random):
    """Copies a random sample of COCO train2017 images (original 0-79 labels,
    untouched) into the merged dataset — the catastrophic-forgetting
    mitigation from Section 6. COCO's own labels already use the exact
    0-79 class scheme this project's classes 0-79 are defined to match, so
    no remapping is needed here, only copying + a fresh train/val split."""
    coco_labels_dir = coco_root / "labels" / "train2017"
    all_labels = sorted(coco_labels_dir.glob("*.txt"))
    if not all_labels:
        print(f"[WARN] No COCO labels found under {coco_labels_dir} — skipping COCO fold-in.")
        return {"train": 0, "val": 0}

    sample = rng.sample(all_labels, k=min(sample_size, len(all_labels)))
    rng.shuffle(sample)
    n_val = max(1, int(len(sample) * val_fraction))
    splits = {"val": sample[:n_val], "train": sample[n_val:]}

    counts = {"train": 0, "val": 0}
    for split, split_labels in splits.items():
        for label_path in split_labels:
            image_candidates = list((coco_root / "images" / "train2017").glob(f"{label_path.stem}.*"))
            image_candidates = [p for p in image_candidates if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not image_candidates:
                continue
            image_path = image_candidates[0]
            shutil.copy2(image_path, out_images / split / image_path.name)
            shutil.copy2(label_path, out_labels / split / label_path.name)
            counts[split] += 1
    return counts


def write_data_yaml(output_dir: Path):
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(LOSTFOUND_CLASS_NAMES)}\n"
        f"names: {LOSTFOUND_CLASS_NAMES!r}\n"
    )
    (output_dir / "lostfound.yaml").write_text(yaml_content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lvis-root", required=True, type=Path, help="Path to Ultralytics' downloaded lvis/ dataset root")
    parser.add_argument("--coco-root", required=True, type=Path, help="Path to Ultralytics' downloaded coco/ dataset root")
    parser.add_argument("--output-dir", default=Path("./lostfound"), type=Path)
    parser.add_argument("--coco-sample-size", type=int, default=750, help="How many COCO train2017 images to fold in (Section 6 recommends 500-1000)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_images, out_labels = args.output_dir / "images", args.output_dir / "labels"
    for split in ("train", "val"):
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    print("Scanning LVIS labels for key/wallet/sunglasses instances...")
    candidates = find_lvis_candidates(args.lvis_root / "labels")
    print(f"Found {len(candidates)} LVIS images containing at least one target class.")
    if not candidates:
        raise SystemExit(
            "No LVIS images matched key/wallet/sunglasses. Check --lvis-root points at "
            "Ultralytics' actual downloaded dataset (it must contain labels/ with .txt files "
            "already converted from LVIS's native COCO-JSON annotations)."
        )

    lvis_counts = copy_lvis_subset(args.lvis_root, candidates, out_images, out_labels, args.val_fraction, rng)
    print(f"Copied LVIS subset: {lvis_counts}")

    coco_counts = sample_coco_fold_in(args.coco_root, args.coco_sample_size, out_images, out_labels, args.val_fraction, rng)
    print(f"Copied COCO fold-in sample: {coco_counts}")

    write_data_yaml(args.output_dir)
    total = lvis_counts["train"] + lvis_counts["val"] + coco_counts["train"] + coco_counts["val"]
    print(f"\nDone. {total} images total. Wrote {args.output_dir / 'lostfound.yaml'}.")
    print(
        "Before training: spot-check ~50-100 of the copied images against their .txt "
        "labels (annotation-quality check from FINE_TUNING_YOLO_GUIDE.md Section 3) — "
        "this script does the mechanical filter/remap/merge correctly, but it cannot "
        "verify LVIS's own box quality for you."
    )


if __name__ == "__main__":
    main()
