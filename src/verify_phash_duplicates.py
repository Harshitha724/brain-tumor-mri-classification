import pandas as pd
import numpy as np
from PIL import Image
import imagehash

df = pd.read_csv("results/clean_split.csv")

train = df[df["original_split"] == "Training"]
test = df[df["original_split"] == "Testing"]

train_hashes = {}

for _, row in train.iterrows():
    img = Image.open(row["path"]).convert("RGB")
    h = imagehash.phash(img)
    train_hashes.setdefault(str(h), []).append(row["path"])

matches = []

for _, row in test.iterrows():
    img = Image.open(row["path"]).convert("RGB")
    h = imagehash.phash(img)

    key = str(h)

    if key in train_hashes:
        for train_path in train_hashes[key]:
            train_img = Image.open(train_path).convert("RGB")
            test_img = img

            train_img = train_img.resize((224, 224))
            test_img = test_img.resize((224, 224))

            a = np.asarray(train_img).astype(np.float32)
            b = np.asarray(test_img).astype(np.float32)

            mae = np.mean(np.abs(a - b))
            mse = np.mean((a - b) ** 2)

            matches.append((mae, mse, train_path, row["path"]))

print(f"pHash=0 train-test pairs: {len(matches)}")

if matches:
    maes = [x[0] for x in matches]
    mses = [x[1] for x in matches]

    print(f"Mean pixel MAE: {np.mean(maes):.4f}")
    print(f"Median pixel MAE: {np.median(maes):.4f}")
    print(f"Maximum pixel MAE: {np.max(maes):.4f}")

    print(f"\nMean pixel MSE: {np.mean(mses):.4f}")

    print("\nExamples:")
    for mae, mse, train_path, test_path in matches[:20]:
        print(f"\nMAE: {mae:.4f}")
        print(f"TRAIN: {train_path}")
        print(f"TEST:  {test_path}")
