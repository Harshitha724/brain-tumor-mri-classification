import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/Training"
TEST_DIR = "data/Testing"


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 4


# ============================================================
# Transformations
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


# ============================================================
# Load datasets
# ============================================================

# Dataset used to obtain all training samples and labels
full_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

# Separate test dataset — NEVER used during training/model selection
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)


# ============================================================
# Stratified Train / Validation Split
# ============================================================

indices = list(range(len(full_dataset)))
labels = full_dataset.targets

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    stratify=labels,
    random_state=RANDOM_SEED
)

train_dataset = Subset(full_dataset, train_indices)

# Validation must NOT use training augmentation
val_full_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=val_test_transform
)

val_dataset = Subset(val_full_dataset, val_indices)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# Dataset information
# ============================================================

classes = full_dataset.classes
class_to_idx = full_dataset.class_to_idx

print("Classes:", classes)
print("Class mapping:", class_to_idx)

print(f"\nTotal training images: {len(full_dataset)}")
print(f"Training images:      {len(train_dataset)}")
print(f"Validation images:    {len(val_dataset)}")
print(f"Test images:          {len(test_dataset)}")