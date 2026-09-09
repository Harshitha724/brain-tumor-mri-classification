from PIL import Image
import imagehash
from dataset import full_dataset, test_dataset

train_hashes = []

for path, label in full_dataset.samples:
    img = Image.open(path).convert("RGB")
    train_hashes.append((imagehash.phash(img), path, label))

test_hashes = []

for path, label in test_dataset.samples:
    img = Image.open(path).convert("RGB")
    test_hashes.append((imagehash.phash(img), path, label))

print(f"Training images: {len(train_hashes)}")
print(f"Test images: {len(test_hashes)}")

for threshold in [0, 2, 5]:
    matches = []

    for h1, path1, label1 in train_hashes:
        for h2, path2, label2 in test_hashes:
            distance = h1 - h2

            if distance <= threshold:
                matches.append((distance, path1, path2, label1, label2))

    print(f"\nThreshold <= {threshold}")
    print(f"Train-test near-duplicate pairs: {len(matches)}")

    for item in matches[:10]:
        print(
            f"  distance={item[0]} | "
            f"{item[1]} <-> {item[2]}"
        )
