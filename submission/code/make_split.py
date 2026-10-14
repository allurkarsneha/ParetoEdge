"""Create a seeded, stratified train/val split of EuroSAT.

The organizers did not provide val_split.json for the software track, so we
generate our own in the same {"train": [...], "val": [...]} format. If the
official file is released later, replace val_split.json and re-run everything.

Usage:
    python make_split.py            # refuses to overwrite an existing file
    python make_split.py --force    # regenerate
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from torchvision import datasets

SEED = 42
VAL_FRACTION = 0.2
OUTPUT = Path("val_split.json")

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="overwrite existing split")
args = parser.parse_args()

if OUTPUT.exists() and not args.force:
    raise SystemExit(f"{OUTPUT} already exists (possibly the official one). Use --force to overwrite.")

# No transform needed: we only read labels, which EuroSAT (an ImageFolder) exposes as .targets
dataset = datasets.EuroSAT(root="./data", download=False)
targets = np.array(dataset.targets)

rng = np.random.default_rng(SEED)
train_indices, val_indices = [], []

# Split each class separately so train and val keep the same class proportions
for class_id in np.unique(targets):
    class_indices = np.flatnonzero(targets == class_id)
    rng.shuffle(class_indices)
    n_val = int(round(len(class_indices) * VAL_FRACTION))
    val_indices.extend(class_indices[:n_val].tolist())
    train_indices.extend(class_indices[n_val:].tolist())

train_indices.sort()
val_indices.sort()

assert not set(train_indices) & set(val_indices), "train/val overlap"
assert len(train_indices) + len(val_indices) == len(dataset), "indices missing"

OUTPUT.write_text(json.dumps({"train": train_indices, "val": val_indices}))

print(f"Total: {len(dataset)} | Train: {len(train_indices)} | Val: {len(val_indices)}")
print(f"Seed: {SEED} | Val fraction: {VAL_FRACTION}")
val_counts = Counter(targets[val_indices].tolist())
print("Val images per class:")
for class_id, class_name in enumerate(dataset.classes):
    print(f"  {class_name}: {val_counts[class_id]}")
print(f"Saved: {OUTPUT}")
