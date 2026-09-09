import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms


# --------------------------------------------------
# Load saved predictions
# --------------------------------------------------

true_labels = np.load("results/test_labels.npy")
predictions = np.load("results/test_predictions.npy")


# --------------------------------------------------
# Load test dataset WITHOUT normalization
# --------------------------------------------------

TEST_DIR = "data/Testing"

display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=display_transform
)

classes = test_dataset.classes

print("Classes:", classes)
print("Total test images:", len(test_dataset))


# --------------------------------------------------
# Find misclassified images
# --------------------------------------------------

misclassified_indices = np.where(
    true_labels != predictions
)[0]

print("\nTotal misclassified images:", len(misclassified_indices))


# --------------------------------------------------
# Count errors by actual class
# --------------------------------------------------

print("\nMisclassified images by actual class:")

for class_index, class_name in enumerate(classes):

    count = np.sum(
        (true_labels == class_index) &
        (predictions != class_index)
    )

    print(f"{class_name:12s}: {count}")


# --------------------------------------------------
# Focus on glioma errors
# --------------------------------------------------

glioma_index = classes.index("glioma")

glioma_errors = np.where(
    (true_labels == glioma_index) &
    (predictions != glioma_index)
)[0]

print("\nGlioma misclassified images:", len(glioma_errors))


# --------------------------------------------------
# Display glioma errors
# --------------------------------------------------

num_images = min(20, len(glioma_errors))

fig, axes = plt.subplots(
    4,
    5,
    figsize=(15, 12)
)

axes = axes.flatten()

for i in range(num_images):

    idx = glioma_errors[i]

    image, _ = test_dataset[idx]

    image = image.permute(1, 2, 0).numpy()

    true_class = classes[true_labels[idx]]
    predicted_class = classes[predictions[idx]]

    axes[i].imshow(image)

    axes[i].set_title(
        f"True: {true_class}\nPred: {predicted_class}"
    )

    axes[i].axis("off")


# Hide unused axes
for i in range(num_images, len(axes)):
    axes[i].axis("off")


plt.suptitle(
    "Misclassified Glioma Images",
    fontsize=16
)

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

os.makedirs("results/plots", exist_ok=True)

output_path = (
    "results/plots/"
    "misclassified_glioma.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved to:")
print(output_path)