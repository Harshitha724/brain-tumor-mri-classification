"""import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader
from models import (
    SimpleCNN,
    get_resnet34_frozen,
    get_resnet34_finetuned
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
LEARNING_RATE = 0.001

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# Training function
# ============================================================

def train_model(model, model_name, epochs=EPOCHS, lr=LEARNING_RATE):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_accuracy = 0.0

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}%"
        )


        # ----------------------------------------------------
        # Save BEST checkpoint
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{model_name}_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path
            )

            print(
                f"  ✓ Best model saved "
                f"(Val Acc: {val_accuracy:.2f}%)"
            )

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    return model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Simple CNN
    # --------------------------------------------------------

    cnn_model = SimpleCNN(num_classes=4)

    train_model(
        cnn_model,
        "simple_cnn"
    )


    # --------------------------------------------------------
    # 2. ResNet34 Frozen
    # --------------------------------------------------------

    resnet_frozen = get_resnet34_frozen(num_classes=4)

    train_model(
        resnet_frozen,
        "resnet34_frozen"
    )


    # --------------------------------------------------------
    # 3. ResNet34 Fine-tuned
    # --------------------------------------------------------

    resnet_finetuned = get_resnet34_finetuned(num_classes=4)

    train_model(
        resnet_finetuned,
        "resnet34_finetuned",
        lr=0.0001
    )"""
    
    
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader
from models import (
    SimpleCNN,
    get_resnet34_frozen,
    get_resnet34_finetuned
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# Training function
# ============================================================

def train_model(model, model_name, epochs=10, lr=0.001):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_accuracy = 0.0

    # Store results from every epoch
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    for epoch in range(epochs):

        # ====================================================
        # TRAINING
        # ====================================================

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


        # ====================================================
        # VALIDATION
        # ====================================================

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


        # ====================================================
        # Save history
        # ====================================================

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)


        # ====================================================
        # Print results
        # ====================================================

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )


        # ====================================================
        # Save BEST checkpoint
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{model_name}_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "history": history
                },
                checkpoint_path
            )

            print(
                f"  ✓ Best model saved "
                f"(Val Acc: {val_accuracy:.2f}%)"
            )

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    return model, history, best_val_accuracy


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Simple CNN
    # --------------------------------------------------------

    cnn_model = SimpleCNN(num_classes=4)

    train_model(
        cnn_model,
        "simple_cnn",
        epochs=10,
        lr=0.001
    )


    # --------------------------------------------------------
    # ResNet34 Frozen
    # --------------------------------------------------------

    resnet_frozen = get_resnet34_frozen(num_classes=4)

    train_model(
        resnet_frozen,
        "resnet34_frozen",
        epochs=10,
        lr=0.001
    )


    # --------------------------------------------------------
    # ResNet34 Fine-tuned
    # --------------------------------------------------------

    resnet_finetuned = get_resnet34_finetuned(num_classes=4)

    train_model(
        resnet_finetuned,
        "resnet34_finetuned",
        epochs=10,
        lr=0.0001
    )