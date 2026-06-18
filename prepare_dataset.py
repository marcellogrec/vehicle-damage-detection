"""
prepare_dataset.py
==================
Extracts the vehicle damage dataset (DOI: 10.7910/DVN/MV5EME) from a ZIP file,
performs a stratified 80% train / 10% validation / 10% test split, and
generates the folder structure expected by YOLO

Generated structure:
    dataset/
    ├── images/
    │   ├── train/   ← 80% of images
    │   ├── val/     ← 10% of images (per-epoch evaluation)
    │   └── test/    ← 10% of images (final evaluation)
    └── labels/
        ├── train/
        ├── val/
        └── test/

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --zip_path other/path.zip --output_dir my_dataset
    python prepare_dataset.py --val_size 0.15 --test_size 0.15
"""

import argparse
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

# DEFAULT SETTINGS
DEFAULT_ZIP    = "damages_dataset.zip"       # file downloaded from Harvard Dataverse
DEFAULT_OUTPUT = "damage_dataset_out"        # root folder of the organised dataset
TEST_SIZE      = 0.10                        # reserved for testing
VAL_SIZE       = 0.10                        # reserved for validation
RANDOM_SEED    = 42                          


ZIP_IMAGES_PREFIX = "damages/damages/images/"
ZIP_LABELS_PREFIX = "damages/damages/labels/"

CLASS_NAMES = ["Scratched", "Dented", "Broken"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepares vehicle damage dataset for YOLO (80/10/10 split)"
    )
    p.add_argument("--zip_path",   default=DEFAULT_ZIP)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    p.add_argument("--val_size",   type=float, default=VAL_SIZE,
                   help="Validation fraction (default: 0.10)")
    p.add_argument("--test_size",  type=float, default=TEST_SIZE,
                   help="Test fraction (default: 0.10)")
    p.add_argument("--seed",       type=int,   default=RANDOM_SEED)
    return p.parse_args()


def get_dominant_class(label_bytes: bytes) -> int:
    """Returns the most frequent class in the label file for stratification."""
    counts: Counter = Counter()
    for line in label_bytes.decode("utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            try:
                counts[int(parts[0])] += 1
            except ValueError:
                pass
    return counts.most_common(1)[0][0] if counts else 0


def split_stratified(pairs: list, val_size: float, test_size: float, seed: int):
    """
    Splits pairs into train / val / test while preserving the dominant-class ratio.
    Returns: (train_pairs, val_pairs, test_pairs)
    """
    rng = random.Random(seed)

    by_class: dict = {}
    for img, lbl, cls in pairs:
        by_class.setdefault(cls, []).append((img, lbl))

    train, val, test = [], [], []
    for cls, items in sorted(by_class.items()):
        rng.shuffle(items)
        n        = len(items)
        n_val    = max(1, round(n * val_size))
        n_test   = max(1, round(n * test_size))
        # ensure val + test do not consume the entire subset
        n_test   = min(n_test,  n - n_val - 1)
        n_val    = min(n_val,   n - n_test - 1)

        val_items   = items[:n_val]
        test_items  = items[n_val : n_val + n_test]
        train_items = items[n_val + n_test:]

        val.extend  ([(img, lbl, cls) for img, lbl in val_items])
        test.extend ([(img, lbl, cls) for img, lbl in test_items])
        train.extend([(img, lbl, cls) for img, lbl in train_items])

    return train, val, test


def prepare(zip_path: str, output_dir: str,
            val_size: float, test_size: float, seed: int):
    zip_path   = Path(zip_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found: {zip_path}\n"
            f"\n\nDownload the dataset at: https://doi.org/10.7910/DVN/MV5EME\n\n"
        )

    train_pct = (1 - val_size - test_size) * 100
    print(f"\n{'='*60}")
    print("Dataset Preparation - Vehicle Damage Detection")
    print(f"{'='*60}")
    print(f"ZIP: {zip_path}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Split: {train_pct:.0f}% train / "
          f"{val_size*100:.0f}% val / {test_size*100:.0f}% test")
    print(f"Seed: {seed}\n")

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        img_stems = {}
        for name in all_names:
            if name.startswith(ZIP_IMAGES_PREFIX) and not name.endswith("/"):
                img_stems[Path(name).stem] = name

        lbl_stems = {}
        for name in all_names:
            if name.startswith(ZIP_LABELS_PREFIX) and name.endswith(".txt"):
                lbl_stems[Path(name).stem] = name

        common_stems = set(img_stems) & set(lbl_stems)
        print(f"Images found: {len(img_stems)}")
        print(f"Labels found: {len(lbl_stems)}")
        print(f"Valid pairs: {len(common_stems)}\n")

        pairs = []
        for stem in common_stems:
            lbl_data = zf.read(lbl_stems[stem])
            dom_cls  = get_dominant_class(lbl_data)
            pairs.append((img_stems[stem], lbl_stems[stem], dom_cls))

        train_pairs, val_pairs, test_pairs = split_stratified(
            pairs, val_size, test_size, seed
        )

        print(f"Train: {len(train_pairs)} pairs"
              f"({len(train_pairs)/len(pairs)*100:.1f}%)")
        print(f"Val:   {len(val_pairs)} pairs"
              f"({len(val_pairs)/len(pairs)*100:.1f}%)")
        print(f"Test:  {len(test_pairs)} pairs"
              f"({len(test_pairs)/len(pairs)*100:.1f}%)\n")

        def extract_pairs(pair_list, split_name: str):
            count = 0
            for img_zip_path, lbl_zip_path, _ in pair_list:
                img_name = Path(img_zip_path).name
                img_dest = output_dir / "images" / split_name / img_name
                with zf.open(img_zip_path) as src, open(img_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                lbl_name = Path(lbl_zip_path).name
                lbl_dest = output_dir / "labels" / split_name / lbl_name
                with zf.open(lbl_zip_path) as src, open(lbl_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                count += 1
                if count % 200 == 0:
                    print(f"    [{split_name}] {count}/{len(pair_list)} copied...",
                          flush=True)

            print(f"[{split_name}]: {count} pairs extracted")

        print("Extracting train...")
        extract_pairs(train_pairs, "train")

        print("Extracting val...")
        extract_pairs(val_pairs, "val")

        print("Extracting test...")
        extract_pairs(test_pairs, "test")

        write_yaml(str(output_dir.resolve()), output_dir / "damages.yaml")

    print_class_stats(train_pairs, val_pairs, test_pairs)
    print(f"\nDataset ready at: {output_dir.resolve()}")

def write_yaml(dataset_path: str, dest: Path):
    content = f"""# damages.yaml
# Dataset: Vehicle Damage Detection
# Source: https://doi.org/10.7910/DVN/MV5EME
# Split: 80% train / 10% val / 10% test

path:  {dataset_path}
train: images/train
val:   images/val
test:  images/test

nc: 3
names:
  0: Scratched   # surface scratch
  1: Dented      # dent / deformation
  2: Broken      # broken part / shattered glass
"""
    dest.write_text(content)


def print_class_stats(train_pairs, val_pairs, test_pairs):
    all_pairs = train_pairs + val_pairs + test_pairs
    print("\nDistribution by dominant class:")
    print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"{'-'*50}")
    for cls_id, name in enumerate(CLASS_NAMES):
        tr = sum(1 for _, _, c in train_pairs if c == cls_id)
        vl = sum(1 for _, _, c in val_pairs   if c == cls_id)
        te = sum(1 for _, _, c in test_pairs  if c == cls_id)
        print(f"  {name:<15} {tr:>8} {vl:>8} {te:>8} {tr+vl+te:>8}")
    print(f"{'TOTAL':<15} {len(train_pairs):>8} {len(val_pairs):>8} "
          f"{len(test_pairs):>8} {len(all_pairs):>8}")


if __name__ == "__main__":
    args = parse_args()
    prepare(
        zip_path   = args.zip_path,
        output_dir = args.output_dir,
        val_size   = args.val_size,
        test_size  = args.test_size,
        seed       = args.seed,
    )