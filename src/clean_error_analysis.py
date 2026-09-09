import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from models import get_resnet34_finetuned
from clean_dataset import test_df, classes

CHECKPOINT = "checkpoints/resnet34_clean_lr5e-5_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

model = get_resnet34_finetuned(num_classes=4)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

errors = []

for _, row in test_df.iterrows():

    image = Image.open(row["path"]).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

    true_label = row["class"]

    if prediction != classes.index(true_label):

        predicted_label = classes[prediction]

        errors.append({
            "path": row["path"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence
        })

errors_df = pd.DataFrame(errors)

print("\n" + "=" * 60)
print("CLEAN TEST ERROR ANALYSIS")
print("=" * 60)

print(f"\nTotal test images: {len(test_df)}")
print(f"Misclassified images: {len(errors_df)}")
print(f"Accuracy: {(1 - len(errors_df)/len(test_df))*100:.2f}%")

print("\nMisclassification breakdown:")
print(
    errors_df.groupby(
        ["true_label", "predicted_label"]
    ).size()
)

print("\nDetailed errors:")
print(errors_df.to_string(index=False))

output_path = "results/error_analysis/clean_misclassified_images.csv"

errors_df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")
