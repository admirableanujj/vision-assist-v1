"""
Per-class instance-count EDA on the merged lostfound dataset — run this
BEFORE spending any Kaggle GPU time training (docs/FINE_TUNING_YOLO_GUIDE.md
Section 3.3). prepare_dataset.py only ever prints IMAGE-level counts (how
many images matched each source) — it never counts how many box INSTANCES
of each of the 48 classes actually survived the merge. A class can appear
in plenty of images but still have very few boxes (e.g. one small object
in a crowded scene), so image counts alone can hide real class imbalance.

Reads the YOLO-format label .txt files prepare_dataset.py already wrote
(one line per box: "<class_id> <x> <y> <w> <h>") and counts lines per
class, train+val combined and separately. Pure counting — no GPU, no
ultralytics import needed, so this runs anywhere Python + prepare_dataset.py
are available (Kaggle or local).

Usage:
    python eda.py --dataset-dir ./lostfound --output-dir ./eda_report
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_dataset import LOSTFOUND_CLASS_NAMES, NUM_KEPT_COCO


def count_instances(labels_dir: Path, num_classes: int) -> tuple[Counter, Counter]:
    """Returns (counts, invalid) — counts only ever holds this project's
    real class ids (0..num_classes-1); anything outside that range (a
    corrupted label, a mis-remapped id) is tracked separately in `invalid`
    rather than silently folded into totals that wouldn't match the
    per-class CSV rows."""
    counts = Counter()
    invalid = Counter()
    for label_path in sorted(labels_dir.rglob("*.txt")):
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                class_id = int(line.split()[0])
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping malformed label line {label_path}:{line_no} ({e!r}): {line!r}")
                continue
            if not (0 <= class_id < num_classes):
                invalid[class_id] += 1
                print(f"[WARN] {label_path}:{line_no} has out-of-range class id {class_id} (valid range 0-{num_classes - 1}) — excluded from counts")
                continue
            counts[class_id] += 1
    return counts, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="prepare_dataset.py's --output-dir (must contain labels/{train,val})")
    parser.add_argument("--output-dir", default=Path("./eda_report"), type=Path)
    parser.add_argument("--imbalance-floor", type=int, default=50, help="Classes with fewer than this many TOTAL instances (train+val) are flagged as under-represented — review before training, not after")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(LOSTFOUND_CLASS_NAMES)
    per_split = {}
    invalid_by_split = {}
    image_counts = {}
    image_exts = {".jpg", ".jpeg", ".png"}
    for split in ("train", "val"):
        per_split[split], invalid_by_split[split] = count_instances(args.dataset_dir / "labels" / split, num_classes)
        image_counts[split] = len([p for p in (args.dataset_dir / "images" / split).glob("*") if p.suffix.lower() in image_exts])

    total_counts = per_split["train"] + per_split["val"]
    total_invalid = invalid_by_split["train"] + invalid_by_split["val"]
    total_images = sum(image_counts.values())
    total_instances = sum(total_counts.values())  # excludes invalid ids, so this always equals the CSV's own total_instances column sum
    total_invalid_instances = sum(total_invalid.values())

    if total_images == 0 and total_instances == 0 and total_invalid_instances == 0:
        raise SystemExit(
            f"Found 0 images and 0 label instances under {args.dataset_dir} — check "
            "--dataset-dir actually points at prepare_dataset.py's --output-dir "
            "(must contain images/{train,val} and labels/{train,val}), not a typo'd "
            "or empty path. An all-zero report below would look identical to a real "
            "empty dataset, so this is checked explicitly rather than silently proceeding."
        )

    rows = [
        {
            "class_id": idx,
            "class_name": name,
            "group": "kept_coco" if idx < NUM_KEPT_COCO else "novel",
            "train_instances": per_split["train"].get(idx, 0),
            "val_instances": per_split["val"].get(idx, 0),
            "total_instances": total_counts.get(idx, 0),
        }
        for idx, name in enumerate(LOSTFOUND_CLASS_NAMES)
    ]
    rows.sort(key=lambda r: -r["total_instances"])

    nonzero = [r["total_instances"] for r in rows if r["total_instances"] > 0]
    zero_classes = [r["class_name"] for r in rows if r["total_instances"] == 0]
    under_floor = [r["class_name"] for r in rows if 0 < r["total_instances"] < args.imbalance_floor]

    summary = {
        "total_images": total_images,
        "total_instances": total_instances,
        "images_by_split": image_counts,
        "min_class_instances": min(nonzero) if nonzero else 0,
        "max_class_instances": max(nonzero) if nonzero else 0,
        "imbalance_ratio_max_over_min": round(max(nonzero) / min(nonzero), 1) if nonzero else None,
        "classes_with_zero_instances": zero_classes,
        f"classes_under_{args.imbalance_floor}_instance_floor": under_floor,
        "invalid_class_id_instances_excluded": total_invalid_instances,
        "invalid_class_ids_seen": sorted(total_invalid.keys()),
    }

    csv_path = args.output_dir / "class_distribution.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # Chart generation is best-effort and must never take down the rest of
    # this script (the CSV/JSON above are already written) — a narrow
    # except ImportError here previously let any OTHER chart-building
    # failure (e.g. a disk-full savefig, a backend quirk) crash the process
    # before the final console summary below ever ran.
    chart_path = args.output_dir / "class_distribution.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("[WARN] matplotlib not installed — skipping chart. CSV/JSON were still written.")

    if plt is not None:
        try:
            names = [r["class_name"] for r in rows]
            counts = [r["total_instances"] for r in rows]
            colors = ["#4C72B0" if r["group"] == "kept_coco" else "#DD8452" for r in rows]
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.bar(names, counts, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=7)
            ax.set_ylabel("Instance count (train+val)")
            ax.set_title("Per-class instance counts — blue = kept COCO (0-19), orange = novel LVIS (20-47)")
            if args.imbalance_floor:
                ax.axhline(args.imbalance_floor, color="red", linestyle="--", linewidth=1, label=f"imbalance floor ({args.imbalance_floor})")
                ax.legend()
            fig.tight_layout()
            fig.savefig(chart_path, dpi=150)
            print(f"Wrote chart: {chart_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate chart ({e!r}) — skipping. CSV/JSON were still written.")

    print(f"\n{total_images} images ({image_counts}), {total_instances} box instances, {len(LOSTFOUND_CLASS_NAMES)} classes.")
    if total_invalid_instances:
        print(f"[WARN] {total_invalid_instances} label line(s) had out-of-range class ids {sorted(total_invalid.keys())} — excluded from the counts above and from the CSV. Check for a corrupted or mis-remapped label file.")
    if zero_classes:
        print(f"[WARN] {len(zero_classes)} class(es) have ZERO instances — training will not learn these at all: {zero_classes}")
    if under_floor:
        print(f"[WARN] {len(under_floor)} class(es) under the {args.imbalance_floor}-instance floor: {under_floor}")
    if nonzero:
        print(f"Imbalance ratio (max/min instances, excluding zero-instance classes): {summary['imbalance_ratio_max_over_min']}x")
    print(f"\nFull per-class breakdown: {csv_path}")
    print("Review this BEFORE training — see docs/FINE_TUNING_YOLO_GUIDE.md Section 3.3 for what to do about flagged classes (raise --coco-sample-size, add a Roboflow fallback for a sparse novel class, or accept and note the limitation).")


if __name__ == "__main__":
    main()
