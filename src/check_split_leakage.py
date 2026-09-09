from PIL import Image
import imagehash
from dataset import train_indices, val_indices, full_dataset

items = []

for idx in range(len(full_dataset)):
    path, label = full_dataset.samples[idx]
    img = Image.open(path).convert("RGB")
    h = imagehash.phash(img)
    split = "train" if idx in set(train_indices) else "val"
    items.append((h, path, label, split))

train_items = [x for x in items if x[3] == "train"]
val_items = [x for x in items if x[3] == "val"]

print(f"Train images: {len(train_items)}")
print(f"Validation images: {len(val_items)}")

for threshold in [0, 2, 5]:
    matches = []

    for h1, path1, label1, _ in train_items:
        for h2, path2, label2, _ in val_items:
            distance = h1 - h2

            if distance <= threshold:
                matches.append((distance, path1, path2, label1, label2))

    print(f"\nThreshold <= {threshold}")
    print(f"Cross-split near-duplicate pairs: {len(matches)}")

    if matches:
        print("Examples:")
        for item in matches[:10]:
            print(
                f"  distance={item[0]} | "
                f"{item[1]} <-> {item[2]}"
            )
