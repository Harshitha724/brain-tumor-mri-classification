import os
import torch
import torch.nn as nn
import torch.optim as optim

from clean_dataset import train_loader, val_loader
from models import get_resnet34_finetuned

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 10
LR = 5e-5


def train_model():

    model = get_resnet34_finetuned(num_classes=4)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_val_accuracy = 0.0

    print("\n" + "=" * 60)
    print("CLEAN LEAKAGE-AWARE RESNET34 TRAINING")
    print(f"Learning Rate: {LR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # ---------------- VALIDATION ----------------
        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_total += loss.item()

                predictions = outputs.argmax(dim=1)

                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "resnet34_clean_lr5e-5_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy
                },
                checkpoint_path
            )

            print(
                f"  ✓ Best clean model saved "
                f"(Val Acc: {val_accuracy:.2f}%)"
            )

    print(
        f"\nBest clean validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )


if __name__ == "__main__":
    train_model()
