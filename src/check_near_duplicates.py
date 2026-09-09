import os
from PIL import Image
import imagehash

TRAIN_DIR = "data/Training"

hashes = []

for root, _, files in os.walk(TRAIN_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            try:
                img = Image.open(path).convert("RGB")
                h = imagehash.phash(img)
                hashes.append((h, path))
            except Exception as e:
                print(f"Could not process {path}: {e}")

print(f"Images processed: {len(hashes)}")

threshold = 5
matches = []

for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        distance = hashes[i][0] - hashes[j][0]

        if distance <= threshold:
            matches.append(
                (distance, hashes[i][1], hashes[j][1])
            )

matches.sort()

print(f"\nNear-duplicate pairs (pHash distance <= {threshold}): {len(matches)}")

for distance, path1, path2 in matches[:100]:
    print(f"\nDistance: {distance}")
    print(f"  {path1}")
    print(f"  {path2}")
