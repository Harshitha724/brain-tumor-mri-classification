import os
import random
import pandas as pd
import imagehash
from PIL import Image, ImageDraw

random.seed(42)

df = pd.read_csv("results/clean_split.csv")

items = []

for _, row in df.iterrows():
    img = Image.open(row["path"]).convert("RGB")
    h = imagehash.phash(img)

    items.append({
        "hash": h,
        "path": row["path"],
        "class": row["class"],
        "split": row["clean_split"]
    })

pairs = []

# Only inspect train-val and train-test crossings
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        a = items[i]
        b = items[j]

        if a["split"] == b["split"]:
            continue

        distance = a["hash"] - b["hash"]

        if distance <= 5:
            pairs.append({
                "distance": distance,
                "path1": a["path"],
                "path2": b["path"],
                "class1": a["class"],
                "class2": b["class"],
                "split1": a["split"],
                "split2": b["split"]
            })

pairs.sort(key=lambda x: x["distance"])

print(f"Cross-split near-duplicate pairs: {len(pairs)}")

# Save pair information
os.makedirs("results/near_duplicate_inspection", exist_ok=True)

pd.DataFrame(pairs).to_csv(
    "results/near_duplicate_inspection/pairs.csv",
    index=False
)

# Select representative pairs
selected = []

seen = set()

for pair in pairs:
    key = tuple(sorted([pair["path1"], pair["path2"]]))

    if key not in seen:
        selected.append(pair)
        seen.add(key)

    if len(selected) >= 40:
        break

print(f"Selected {len(selected)} representative pairs.")

# Create contact sheets
thumb_size = (180, 180)
sheet_width = 800
sheet_height = 430

for idx, pair in enumerate(selected):

    img1 = Image.open(pair["path1"]).convert("RGB")
    img2 = Image.open(pair["path2"]).convert("RGB")

    img1.thumbnail(thumb_size)
    img2.thumbnail(thumb_size)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    sheet.paste(img1, (50, 50))
    sheet.paste(img2, (350, 50))

    draw.text(
        (50, 250),
        f"{pair['split1']} | {pair['class1']}",
        fill="black"
    )

    draw.text(
        (350, 250),
        f"{pair['split2']} | {pair['class2']}",
        fill="black"
    )

    draw.text(
        (50, 290),
        f"pHash distance: {pair['distance']}",
        fill="black"
    )

    draw.text(
        (50, 320),
        os.path.basename(pair["path1"]),
        fill="black"
    )

    draw.text(
        (50, 350),
        os.path.basename(pair["path2"]),
        fill="black"
    )

    output = (
        f"results/near_duplicate_inspection/"
        f"pair_{idx:03d}_d{pair['distance']}.png"
    )

    sheet.save(output)

print("\nSaved:")
print("  results/near_duplicate_inspection/pairs.csv")
print("  40 representative image-pair files")
