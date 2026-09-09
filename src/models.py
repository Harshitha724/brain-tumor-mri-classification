import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 1. Simple CNN
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 224 → 112 → 56 → 28 → 14
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.fc2(x)

        return x


# ============================================================
# 2. ResNet34 - Frozen Backbone
# ============================================================

def get_resnet34_frozen(num_classes=4):

    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1
    )

    # Freeze the entire pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model


# ============================================================
# 3. ResNet34 - Fully Fine-Tuned
# ============================================================

def get_resnet34_finetuned(num_classes=4):

    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1
    )

    # All layers remain trainable
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model