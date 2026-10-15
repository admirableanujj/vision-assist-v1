"""
Build the lostfound/ training dataset for a curated, use-case-specific class
list — NOT the original "80 COCO classes + 3 novel" scheme. See
docs/FINE_TUNING_YOLO_GUIDE.md Section 2 for the full rationale: most of
COCO's 80 classes (wild animals, street infrastructure, vehicles, food,
furniture/appliances) are irrelevant to "find a misplaced item in a living
room, home, or garage" and were dropped; a curated set of 20 still-relevant
COCO classes was kept, and 28 new classes for things people actually search
for at home/in the garage were added, sourced from LVIS (verified against
its real class list, not guessed).

Two sources are merged:
  - LVIS images containing any of the 28 novel classes (remapped to indices
    20-47) — always included in full.
  - A capped random sample of COCO train2017 images containing any of the
    20 kept COCO classes (remapped to indices 0-19) — the catastrophic-
    forgetting mitigation from Section 6, now scoped to only the classes
    actually kept, not all 80.

LVIS reuses COCO's own train2017/val2017 JPEGs verbatim, so an image can
legitimately match both sources (e.g. a photo with both a "person" and a
"key" in frame) — such images get BOTH classes' lines merged into one label
file rather than one source's lines overwriting the other's.

Run on Kaggle, inside the training notebook, AFTER Ultralytics has already
populated its standard LVIS and COCO dataset layout on disk — this script
only filters/remaps/merges, it doesn't download either dataset itself.
Trigger that first:

    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")
    model.train(data="lvis.yaml", epochs=1)  # downloads + converts to YOLO format, then stop
    model.train(data="coco.yaml", epochs=1)  # same, for COCO

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
    ./lostfound/labels/{train,val}/*.txt   # remapped to this project's 48-class scheme
    ./lostfound/lostfound.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

# The 80 standard COCO classes, in the exact order every COCO-pretrained YOLO
# checkpoint uses — needed only to compute each kept class's ORIGINAL index,
# since that's what COCO's own downloaded label files use.
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

# 20 of the 80 COCO classes kept — small, portable, personally-owned items
# plausible to search for in a living room/home/garage. Everything dropped
# (wild animals, street infra, vehicles, food, furniture/appliances) is
# either a fixture that's never really "lost", or outside this product's
# scope. New final indices 0-19, in this order.
KEPT_COCO_CLASS_NAMES = [
    "person", "bicycle", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "bottle", "cup", "book", "scissors", "teddy bear", "toothbrush", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "cat", "dog",
]
assert set(KEPT_COCO_CLASS_NAMES) <= set(COCO_80_CLASSES)

# Maps each kept class's COCO label-file index (0-79) to its new index
# (0-19) in this project's scheme.
COCO_OLD_TO_NEW = {COCO_80_CLASSES.index(name): i for i, name in enumerate(KEPT_COCO_CLASS_NAMES)}

# 28 new classes for things actually searched for at home or in a garage —
# NOT in COCO's 80. Each (name, lvis_class_id) pair confirmed directly
# against Ultralytics' own lvis.yaml (v8.4.106, matching this repo's pin) —
# every id below is a real, verified LVIS class, not a guess. New final
# indices 20-47, in this order (20 + position).
#
# Deliberately excluded despite existing in LVIS: fine-grained jewelry
# (ring/earring/necklace/bracelet — LVIS ids 884/410/714/145) — too small
# and reflective for reliable webcam-resolution detection, and would bloat
# annotation effort disproportionately for likely-poor accuracy; flagged as
# a stretch goal, not silently dropped. Footwear/headwear subtypes (boot,
# sandal, slipper, flip-flop, cowboy hat, bowler hat, baseball cap, etc.)
# are consolidated into single "shoe" and "hat" classes rather than kept as
# separate LVIS subtypes, to avoid class-imbalance/confusion between near-
# identical categories a nano model has little hope of telling apart.
NOVEL_CLASSES = [
    ("keys", 604),
    ("wallet", 1155),
    ("sunglasses", 1034),
    ("glasses", 994),           # spectacles/specs/eyeglasses/glasses
    ("watch", 1160),            # watch/wristwatch
    ("earphone", 408),          # earphone/earpiece/headphone
    ("flashlight", 456),        # flashlight/torch
    ("glove", 499),
    ("belt", 87),
    ("pen", 780),
    ("notebook", 723),
    ("calculator", 183),
    ("stapler", 1005),          # stapler/stapler stapling machine
    ("battery", 69),
    ("shoe", 947),              # shoe/sneaker/tennis shoe
    ("jacket", 588),
    ("briefcase", 152),
    ("passport", 769),
    ("scarf", 920),
    ("hat", 543),
    ("screwdriver", 925),
    ("wrench", 1195),           # wrench/spanner
    ("pliers", 820),
    ("hammer", 529),
    ("toolbox", 1100),
    ("tape measure", 1061),     # tape measure/measuring tape
    ("ladder", 620),
    ("game controller", 598),   # joystick
]
NUM_KEPT_COCO = len(KEPT_COCO_CLASS_NAMES)
LVIS_TO_LOSTFOUND = {lvis_id: NUM_KEPT_COCO + i for i, (_name, lvis_id) in enumerate(NOVEL_CLASSES)}

LOSTFOUND_CLASS_NAMES = KEPT_COCO_CLASS_NAMES + [name for name, _lvis_id in NOVEL_CLASSES]
assert len(LOSTFOUND_CLASS_NAMES) == NUM_KEPT_COCO + len(NOVEL_CLASSES)


def _filter_and_remap(label_path: Path, id_map: dict[int, int]) -> list[str]:
    """
    Reads one YOLO-format label file and returns only the lines whose
    source class id is a key in id_map, rewritten to id_map's value.

    Critical correctness point: both LVIS (1203 classes) and COCO (80
    classes, of which only 20 are kept here) have far more classes than
    this project uses. Lines for everything else must be dropped entirely,
    not left alone — surviving untouched, they'd be silently misread under
    the new, much smaller label space (e.g. original COCO class 42 ==
    "banana", dropped here, would otherwise collide with whatever this
    project's own class 42 happens to be).
    """
    kept = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        source_class_id = int(parts[0])
        if source_class_id in id_map:
            kept.append(f"{id_map[source_class_id]} {' '.join(parts[1:])}")
    return kept


def _find_image(dataset_root: Path, images_subdir: Path, stem: str) -> Path | None:
    candidates = [p for p in images_subdir.glob(f"{stem}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return candidates[0] if candidates else None


def find_candidates(labels_dir: Path, images_root: Path, id_map: dict[int, int]) -> dict[str, dict]:
    """Scans every label file under labels_dir, recursively. Callers MUST
    scope labels_dir to a single split (train2017/ only) for both LVIS and
    COCO — val2017 must stay untouched by training, reserved for
    prepare_coco_forgetting_check.py's "untouched" eval set. Since LVIS
    reuses COCO's own JPEGs verbatim, scanning LVIS's val2017 here too
    would leak those same images into both the training set and the
    supposedly-untouched forgetting-check sample. images_root must mirror
    labels_dir's own subdirectory depth exactly — each label's image is
    located at images_root / <label's path relative to labels_dir>'s
    parent, so the two roots have to line up one-for-one.

    Returns {stem: {"lines": [...], "image_path": Path}} only for images
    containing at least one class in id_map, with lines already filtered/
    remapped via _filter_and_remap."""
    candidates = {}
    for label_path in sorted(labels_dir.rglob("*.txt")):
        kept_lines = _filter_and_remap(label_path, id_map)
        if not kept_lines:
            continue
        rel = label_path.relative_to(labels_dir)
        image_path = _find_image(images_root, images_root / rel.parent, label_path.stem)
        if image_path is None:
            print(f"[WARN] No image found for label {label_path}, skipping")
            continue
        candidates[label_path.stem] = {"lines": kept_lines, "image_path": image_path}
    return candidates


def merge_and_sample(lvis_candidates: dict, coco_candidates: dict, coco_sample_size: int, rng: random.Random) -> dict[str, dict]:
    """Combines both candidate pools by image stem. Every LVIS (novel-class)
    candidate is always included — that's the whole point, these are the
    rare classes. COCO (kept-class) candidates are capped to
    coco_sample_size, sampled from whichever aren't already pulled in via
    LVIS. A stem present in BOTH pools (LVIS reuses COCO's own JPEGs, so
    this happens whenever one photo has both a kept-class and a novel-class
    object) gets both sources' lines merged into one combined label —
    rather than one source silently overwriting the other, which was the
    original version of this script's actual bug."""
    coco_only_stems = [s for s in coco_candidates if s not in lvis_candidates]
    sampled_coco_only = set(rng.sample(coco_only_stems, k=min(coco_sample_size, len(coco_only_stems))))

    final_stems = set(lvis_candidates) | sampled_coco_only
    merged = {}
    for stem in final_stems:
        lines = lvis_candidates.get(stem, {}).get("lines", []) + coco_candidates.get(stem, {}).get("lines", [])
        image_path = lvis_candidates.get(stem, {}).get("image_path") or coco_candidates.get(stem, {}).get("image_path")
        merged[stem] = {"lines": lines, "image_path": image_path}
    return merged


def copy_subset(merged: dict, out_images: Path, out_labels: Path, val_fraction: float, rng: random.Random) -> dict[str, int]:
    """Writes the final merged candidate set into out_images/{train,val}
    and out_labels/{train,val}, assigning the split fresh here rather than
    trusting either source dataset's own split (FINE_TUNING_YOLO_GUIDE.md
    Section 3's "re-split after merging" guidance)."""
    items = list(merged.items())
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_fraction))
    splits = {"val": items[:n_val], "train": items[n_val:]}

    counts = {"train": 0, "val": 0}
    for split, split_items in splits.items():
        for _stem, entry in split_items:
            image_path = entry["image_path"]
            dest_image = out_images / split / image_path.name
            dest_label = out_labels / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dest_image)
            dest_label.write_text("\n".join(entry["lines"]) + "\n")
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
    parser.add_argument("--coco-sample-size", type=int, default=750, help="Max COCO train2017 images to fold in for the 20 kept classes, beyond whatever LVIS candidates already cover (Section 6 recommends 500-1000)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_images, out_labels = args.output_dir / "images", args.output_dir / "labels"
    for split in ("train", "val"):
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    print(f"Scanning LVIS train2017 labels for the {len(NOVEL_CLASSES)} novel classes...")
    # Scoped to train2017 only, same as the COCO call below — NOT the whole
    # labels/ tree. LVIS reuses COCO's own train2017/val2017 JPEGs verbatim,
    # and prepare_coco_forgetting_check.py separately draws its "untouched"
    # forgetting-check sample from COCO's val2017. Scanning LVIS's val2017
    # here too would let an image containing both a novel class (pulled in
    # via LVIS) and a kept class (pulled in via the forgetting-check script)
    # leak into both the training set and the "untouched" eval set at once —
    # exactly the overlap this script already handles for train2017 via
    # merge_and_sample(), but silently unguarded for val2017.
    lvis_candidates = find_candidates(args.lvis_root / "labels" / "train2017", args.lvis_root / "images" / "train2017", LVIS_TO_LOSTFOUND)
    print(f"Found {len(lvis_candidates)} LVIS images containing at least one novel class.")
    if not lvis_candidates:
        raise SystemExit(
            "No LVIS images matched any novel class. Check --lvis-root points at "
            "Ultralytics' actual downloaded dataset (it must contain labels/ with .txt files "
            "already converted from LVIS's native COCO-JSON annotations)."
        )

    print(f"Scanning COCO train2017 labels for the {NUM_KEPT_COCO} kept classes...")
    # labels_dir and images_root must mirror the same subdirectory depth —
    # both scoped to train2017 only here (val2017 stays untouched, reserved
    # for the Section 6 forgetting-check set) — see find_candidates' use of
    # relative_to(labels_dir) to locate each label's matching image.
    coco_candidates = find_candidates(args.coco_root / "labels" / "train2017", args.coco_root / "images" / "train2017", COCO_OLD_TO_NEW)
    print(f"Found {len(coco_candidates)} COCO images containing at least one kept class.")

    merged = merge_and_sample(lvis_candidates, coco_candidates, args.coco_sample_size, rng)
    overlap = set(lvis_candidates) & set(coco_candidates)
    print(f"{len(overlap)} images matched both sources (merged into one combined label each).")

    counts = copy_subset(merged, out_images, out_labels, args.val_fraction, rng)
    print(f"Copied merged dataset: {counts}")

    write_data_yaml(args.output_dir)
    total = counts["train"] + counts["val"]
    print(f"\nDone. {total} images total, {len(LOSTFOUND_CLASS_NAMES)} classes. Wrote {args.output_dir / 'lostfound.yaml'}.")
    print(
        "Before training: spot-check ~50-100 of the copied images against their .txt "
        "labels (annotation-quality check from FINE_TUNING_YOLO_GUIDE.md Section 3) — "
        "this script does the mechanical filter/remap/merge correctly, but it cannot "
        "verify LVIS's/COCO's own box quality for you."
    )


if __name__ == "__main__":
    main()
