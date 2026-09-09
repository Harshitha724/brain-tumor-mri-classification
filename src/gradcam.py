import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import get_resnet34_finetuned


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DIR = "data/Testing"
CHECKPOINT = "checkpoints/resnet34_lr5e-5_best.pth"

OUTPUT_DIR = "results/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Classes
# --------------------------------------------------

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = get_resnet34_finetuned(num_classes=4)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Test dataset
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


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
# Find glioma images
# --------------------------------------------------

glioma_index = classes.index("glioma")

correct_glioma = np.where(
    (true_labels == glioma_index) &
    (predictions == glioma_index)
)[0]

incorrect_glioma = np.where(
    (true_labels == glioma_index) &
    (predictions != glioma_index)
)[0]


print("\nCorrect glioma images:", len(correct_glioma))
print("Incorrect glioma images:", len(incorrect_glioma))


# --------------------------------------------------
# Grad-CAM target layer
# --------------------------------------------------

target_layers = [
    model.layer4[-1]
]


# --------------------------------------------------
# Original image dataset
# --------------------------------------------------

original_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)


# --------------------------------------------------
# Grad-CAM function
# --------------------------------------------------

def generate_gradcam(index, folder_name):

    image_tensor, true_label = test_dataset[index]

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)

    predicted_label = output.argmax(dim=1).item()

    # Original image
    original_image, _ = original_dataset[index]

    rgb_image = original_image.permute(
        1, 2, 0
    ).numpy()

    # Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(predicted_label)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # Save
    true_class = classes[true_label]
    predicted_class = classes[predicted_label]

    filename = (
        f"{index:04d}_"
        f"true_{true_class}_"
        f"pred_{predicted_class}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        folder_name + "_" + filename
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(visualization)

    plt.title(
        f"True: {true_class} | "
        f"Predicted: {predicted_class}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# --------------------------------------------------
# Correct glioma examples
# --------------------------------------------------

print("\nGenerating Grad-CAM for correct glioma examples...")

for index in correct_glioma[:5]:

    generate_gradcam(
        index,
        "correct"
    )


# --------------------------------------------------
# Incorrect glioma examples
# --------------------------------------------------

print("\nGenerating Grad-CAM for incorrect glioma examples...")

for index in incorrect_glioma[:5]:

    generate_gradcam(
        index,
        "incorrect"
    )


print("\nGrad-CAM analysis complete.")
print("Results saved to:", OUTPUT_DIR)
