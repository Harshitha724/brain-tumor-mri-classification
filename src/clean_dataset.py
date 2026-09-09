import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SPLIT_FILE = "results/grouped_clean_split.csv"

BATCH_SIZE = 32
NUM_WORKERS = 4

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        classes = sorted(self.df["class"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")
        label = self.class_to_idx[row["class"]]

        if self.transform:
            image = self.transform(image)

        return image, label


df = pd.read_csv(SPLIT_FILE)

# Remove the 3 conflicting images
df = df[df["clean_split"] != "excluded"].copy()

train_df = df[df["clean_split"] == "train"]
val_df = df[df["clean_split"] == "val"]
test_df = df[df["clean_split"] == "test"]

train_dataset = BrainTumorDataset(train_df, train_transform)
val_dataset = BrainTumorDataset(val_df, val_test_transform)
test_dataset = BrainTumorDataset(test_df, val_test_transform)

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

classes = sorted(df["class"].unique())

print("Classes:", classes)
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")
