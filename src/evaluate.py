import os
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import test_loader, classes
from models import get_resnet34_finetuned


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = get_resnet34_finetuned(num_classes=4)

checkpoint_path = "checkpoints/resnet34_lr5e-5_best.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)

print("\nLoaded checkpoint:")
print("Epoch:", checkpoint["epoch"])
print("Validation Accuracy:", checkpoint["val_accuracy"])


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

model.eval()

test_loss = 0.0
all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


# --------------------------------------------------
# Metrics
# --------------------------------------------------

test_loss = test_loss / len(test_loader)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1 Score:       {f1:.4f}")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# --------------------------------------------------
# Save predictions for later analysis
# --------------------------------------------------

import numpy as np

os.makedirs("results", exist_ok=True)

np.save(
    "results/test_labels.npy",
    np.array(all_labels)
)

np.save(
    "results/test_predictions.npy",
    np.array(all_predictions)
)

np.save(
    "results/confusion_matrix.npy",
    cm
)

print("\nSaved:")
print("  results/test_labels.npy")
print("  results/test_predictions.npy")
print("  results/confusion_matrix.npy")