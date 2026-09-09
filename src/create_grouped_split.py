import os
import random
import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict

SEED = 42
random.seed(SEED)

INPUT = "results/clean_split.csv"
OUTPUT = "results/grouped_clean_split.csv"

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# Create pHash groups
# ---------------------------------------------------------

groups = defaultdict(list)

for idx, row in df.iterrows():
    img = Image.open(row["path"]).convert("RGB")
    h = str(imagehash.phash(img))
    groups[h].append(idx)

print(f"Total images: {len(df)}")
print(f"Unique pHash groups: {len(groups)}")

# ---------------------------------------------------------
# Remove groups containing conflicting labels
# ---------------------------------------------------------

valid_groups = []
conflicting_groups = []

for h, indices in groups.items():

    labels = set(df.loc[indices, "class"])

    if len(labels) > 1:
        conflicting_groups.append(indices)
    else:
        valid_groups.append(indices)

print(f"Conflicting groups removed: {len(conflicting_groups)}")

# Mark excluded images
excluded = set()

for group in conflicting_groups:
    excluded.update(group)

df["clean_split"] = "excluded"

# ---------------------------------------------------------
# Assign groups by class
# ---------------------------------------------------------

class_groups = defaultdict(list)

for group in valid_groups:
    label = df.loc[group[0], "class"]
    class_groups[label].append(group)

for class_name in sorted(class_groups):

    groups_for_class = class_groups[class_name]

    random.shuffle(groups_for_class)

    n = len(groups_for_class)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    for i, group in enumerate(groups_for_class):

        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for idx in group:
            df.loc[idx, "clean_split"] = split

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

os.makedirs("results", exist_ok=True)

df.to_csv(OUTPUT, index=False)

print("\nFinal grouped split:")
print(df["clean_split"].value_counts())

print("\nClass distribution:")
print(
    pd.crosstab(
        df[df["clean_split"] != "excluded"]["class"],
        df[df["clean_split"] != "excluded"]["clean_split"]
    )
)

print("\nExcluded images:", (df["clean_split"] == "excluded").sum())

print(f"\nSaved to: {OUTPUT}")
