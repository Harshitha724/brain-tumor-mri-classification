import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from models import get_resnet34_finetuned
from clean_dataset import test_loader, classes

CHECKPOINT = "checkpoints/resnet34_clean_lr5e-5_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_resnet34_finetuned(num_classes=4)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

cm = confusion_matrix(all_labels, all_predictions)

print("\nClean Test Confusion Matrix:")
print(cm)

print("\nClass order:")
print(classes)

accuracy = np.trace(cm) / np.sum(cm)
print(f"\nAccuracy from confusion matrix: {accuracy * 100:.2f}%")

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ResNet34 Clean Test Confusion Matrix")

plt.tight_layout()

output_path = "results/confusion_matrices/resnet34_clean_confusion_matrix.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nSaved to: {output_path}")

np.save("results/clean_test_labels.npy", all_labels)
np.save("results/clean_test_predictions.npy", all_predictions)

print("Saved clean test labels and predictions.")
