import hashlib
import os

from dataset import train_indices, val_indices, full_dataset

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

train_hashes = {}
val_hashes = {}

for idx in train_indices:
    path, _ = full_dataset.samples[idx]
    train_hashes[get_hash(path)] = path

for idx in val_indices:
    path, _ = full_dataset.samples[idx]
    val_hashes[get_hash(path)] = path

duplicates = set(train_hashes) & set(val_hashes)

print(f"Train images: {len(train_hashes)}")
print(f"Validation images: {len(val_hashes)}")
print(f"Exact duplicates between train and validation: {len(duplicates)}")

if duplicates:
    print("\nDuplicate images found:")
    for h in duplicates:
        print("TRAIN:", train_hashes[h])
        print("VAL:  ", val_hashes[h])
else:
    print("\nNo exact duplicate images found.")
