import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from clean_dataset import test_loader, classes
from models import get_resnet34_finetuned

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/resnet34_clean_lr5e-5_best.pth"

model = get_resnet34_finetuned(num_classes=4)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()

all_labels = []
all_predictions = []

test_loss = 0.0

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

test_loss /= len(test_loader)

accuracy = accuracy_score(all_labels, all_predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("CLEAN TEST EVALUATION")
print("=" * 60)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
print(f"Test images: {len(all_labels)}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1: {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)
