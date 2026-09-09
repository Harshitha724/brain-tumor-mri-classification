import os
import hashlib
import random
import pandas as pd
from collections import defaultdict

DATA_DIR = "data"
OUTPUT = "results/clean_split.csv"

random.seed(42)

records = []

for split in ["Training", "Testing"]:
    split_dir = os.path.join(DATA_DIR, split)

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(class_dir, filename)

                with open(path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                records.append({
                    "path": path,
                    "class": class_name,
                    "original_split": split,
                    "hash": file_hash
                })

df = pd.DataFrame(records)

print(f"Total images: {len(df)}")

# Group exact duplicate files together
groups = defaultdict(list)

for idx, row in df.iterrows():
    groups[row["hash"]].append(idx)

print(f"Unique image groups: {len(groups)}")
print(f"Duplicate groups: {sum(len(v) > 1 for v in groups.values())}")

# Assign each duplicate group to one split.
# We keep the original Testing images preferentially as test groups
# unless the group also contains Training images.
group_records = []

for group_id, indices in enumerate(groups.values()):

    group_df = df.loc[indices]

    classes = group_df["class"].unique()

    # If exact duplicates have conflicting labels, flag them.
    if len(classes) > 1:
        print("\nWARNING: conflicting labels in duplicate group:")
        print(group_df[["path", "class", "original_split"]].to_string(index=False))

    # Majority class for stratification
    label = group_df["class"].mode()[0]

    group_records.append({
        "group_id": group_id,
        "label": label,
        "indices": indices
    })

# Stratified group assignment within each class
assignments = {}

for class_name in sorted(df["class"].unique()):

    class_groups = [
        g for g in group_records
        if g["label"] == class_name
    ]

    random.shuffle(class_groups)

    n = len(class_groups)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    for i, group in enumerate(class_groups):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for idx in group["indices"]:
            assignments[idx] = split

df["clean_split"] = df.index.map(assignments)

os.makedirs("results", exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nClean split created:")
print(df["clean_split"].value_counts())

print("\nClass distribution:")
print(pd.crosstab(df["class"], df["clean_split"]))

print(f"\nSaved to: {OUTPUT}")
