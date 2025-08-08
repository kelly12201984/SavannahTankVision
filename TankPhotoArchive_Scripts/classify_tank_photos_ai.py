# SavannahTankVision – AI Photo Classifier
# -----------------------------------------
# Author: Kelly Kells | Data Scientist & Internal Workflow Engineer
#
# Purpose:
# Scan Savannah Tank’s P: drive job folders, locate shipping photo folders,
# classify images using ML (full-tank vs. detail), and archive full-tank shots
# into a centralized, organized folder — without altering originals.
import os
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
import shutil

# === CONFIG ===
MODEL_PATH = Path(r"P:\TankPhotoArchive\training_sets\tank_classifier_cnn.pth")
ARCHIVE_ROOT = Path(r"P:\TankPhotoArchive\dry_run_archive")
DEST_FULL = Path(r"P:\TankPhotoArchive\classified\full_tanks")
DEST_DETAIL = Path(r"P:\TankPhotoArchive\classified\detail_shots")
IMG_SIZE = 224

for folder in [DEST_FULL, DEST_DETAIL]:
    folder.mkdir(parents=True, exist_ok=True)

# === MODEL LOADING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# === TRANSFORMS ===
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
    return pred


def classify_and_copy(img_path, existing_filenames):
    try:
        pred = predict_image(img_path)

        if pred != 1:
            return

        if img_path.name in existing_filenames:
            print(f"[SKIPPED - ALREADY EXISTS] {img_path.name}")
            return

        shutil.copy2(img_path, DEST_FULL / img_path.name)
        existing_filenames.add(img_path.name)  # Add so next checks are instant
        print(f"[FULL_TANK] {img_path.name}")

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")


def main():
    # List available years/folders
    subfolders = sorted([f for f in ARCHIVE_ROOT.iterdir() if f.is_dir()])
    print("Available years/folders:")
    for idx, f in enumerate(subfolders):
        print(f"{idx+1}: {f.name}")

    while True:
        try:
            choice = input(
                f"Enter numbers of folders to classify, separated by spaces (e.g. 1 3 5): "
            ).split()
            indices = [
                int(c) - 1
                for c in choice
                if c.isdigit() and 1 <= int(c) <= len(subfolders)
            ]
            if not indices:
                print("No valid selection. Try again.")
                continue
            chosen_folders = [subfolders[i] for i in indices]
            break
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")

    # Now, chosen_folders IS defined. Add this next line:
    existing_filenames = set(f.name for f in DEST_FULL.glob("*.jpg"))

    for chosen_folder in chosen_folders:
        images = list(chosen_folder.glob("*.jpg"))
        print(f"\nClassifying {len(images)} images in {chosen_folder}...\n")
        for img_path in images:
            classify_and_copy(img_path, existing_filenames)

    print("\n==== DONE ====")
    print(f"Results in:\n- {DEST_FULL}\n- {DEST_DETAIL}")


if __name__ == "__main__":
    main()
