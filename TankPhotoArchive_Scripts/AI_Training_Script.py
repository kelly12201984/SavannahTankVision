import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----- CONFIG -----
DATA_ROOT = Path(r"P:\TankPhotoArchive\training_sets")

train_dirs = [DATA_ROOT / "full_tanks_train", DATA_ROOT / "detail_shots_train"]
BATCH_SIZE = 16
EPOCHS = 8  # More epochs = better, but 8 is usually enough for transfer learning
MODEL_PATH = DATA_ROOT / "tank_classifier_cnn.pth"
IMG_SIZE = 224  # Required by ResNet

train_dirs = [DATA_ROOT / "full_tanks_train", DATA_ROOT / "detail_shots_train"]

assert all(d.exists() for d in train_dirs), "One or more training folders do not exist!"

# ----- TRANSFORMS -----
train_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ----- DATA -----
dataset = datasets.ImageFolder(DATA_ROOT, transform=train_tfms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Class mapping:", dataset.class_to_idx)

# ----- MODEL -----
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

# ----- TRAIN -----
print(f"Training on {len(dataset)} images ({len(dataset.targets)})...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()

    avg_loss = running_loss / len(dataset)
    acc = correct / len(dataset)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2%}")

# ----- SAVE -----
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# ----- QUICK TEST -----
model.eval()
with torch.no_grad():
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)
    pred = out.argmax(dim=1)
    acc = (pred == yb).float().mean().item()
    print(f"Quick test batch accuracy: {acc:.2%}")
