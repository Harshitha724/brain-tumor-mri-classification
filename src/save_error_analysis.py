import os
import csv
import numpy as np
from torchvision import datasets


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

TEST_DIR = "data/Testing"

test_dataset = datasets.ImageFolder(TEST_DIR)

classes = test_dataset.classes


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

true_labels = np.load(
    "results/test_labels.npy"
)

predictions = np.load(
    "results/test_predictions.npy"
)


# --------------------------------------------------
# Find misclassified images
# --------------------------------------------------

misclassified_indices = np.where(
    true_labels != predictions
)[0]

print("Total misclassified images:", len(misclassified_indices))


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    "results/error_analysis",
    exist_ok=True
)

csv_path = (
    "results/error_analysis/"
    "misclassified_images.csv"
)


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "index",
        "image_path",
        "image_name",
        "true_class",
        "predicted_class"
    ])

    for idx in misclassified_indices:

        image_path = test_dataset.samples[idx][0]

        image_name = os.path.basename(
            image_path
        )

        true_class = classes[
            true_labels[idx]
        ]

        predicted_class = classes[
            predictions[idx]
        ]

        writer.writerow([
            idx,
            image_path,
            image_name,
            true_class,
            predicted_class
        ])


print("\nCSV saved successfully:")
print(csv_path)
