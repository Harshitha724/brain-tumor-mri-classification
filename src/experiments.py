"""import torch

from models import get_resnet34_finetuned
from train import train_model


# ============================================================
# Device
# ============================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Experiments
# ============================================================

experiments = [

    {
        "name": "resnet34_lr5e-5",
        "lr": 5e-5,
        "epochs": 10
    },

    {
        "name": "resnet34_lr1e-5",
        "lr": 1e-5,
        "epochs": 10
    },

    {
        "name": "resnet34_15epochs",
        "lr": 1e-4,
        "epochs": 15
    }

]


# ============================================================
# Run experiments
# ============================================================

results = []


for experiment in experiments:

    print("\n\n")
    print("#" * 70)
    print(f"EXPERIMENT: {experiment['name']}")
    print("#" * 70)

    model = get_resnet34_finetuned(num_classes=4)

    _, history, best_val_accuracy = train_model(
        model,
        experiment["name"],
        epochs=experiment["epochs"],
        lr=experiment["lr"]
    )

    results.append({
        "experiment": experiment["name"],
        "learning_rate": experiment["lr"],
        "epochs": experiment["epochs"],
        "best_val_accuracy": best_val_accuracy
    })


# ============================================================
# Print comparison
# ============================================================

print("\n\n")
print("=" * 70)
print("EXPERIMENT RESULTS")
print("=" * 70)

for result in results:

    print(
        f"{result['experiment']:25s} | "
        f"LR: {result['learning_rate']:<8} | "
        f"Epochs: {result['epochs']:2d} | "
        f"Best Val Acc: {result['best_val_accuracy']:.2f}%"
    )"""
    
from models import get_resnet34_frozen
from train import train_model

print("CUDA available:", __import__("torch").cuda.is_available())

experiments = [
    {
        "name": "resnet34_frozen_lr1e-4",
        "lr": 1e-4,
        "epochs": 10
    }
]

for experiment in experiments:
    print("\n" + "#" * 70)
    print(f"EXPERIMENT: {experiment['name']}")
    print("#" * 70)

    model = get_resnet34_frozen(num_classes=4)

    _, history, best_val_accuracy = train_model(
        model,
        experiment["name"],
        epochs=experiment["epochs"],
        lr=experiment["lr"]
    )

    print(
        f"\nFinal result: "
        f"{experiment['name']} | "
        f"Best Val Accuracy: {best_val_accuracy:.2f}%"
    )