import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import shutil

# ---- CONFIG ----
MODEL_PATH = r"P:\TankPhotoArchive\TankPhotoArchive_Scripts\tank_classifier_knn.pkl"
ARCHIVE_ROOT = r"P:\TankPhotoArchive\dry_run_archive"
DEST_FULL = r"P:\TankPhotoArchive\classified\full_tanks"
DEST_DETAIL = r"P:\TankPhotoArchive\classified\detail_shots"
HIST_BINS = 32

# ---- Ensure output folders exist ----
for folder in [DEST_FULL, DEST_DETAIL, DEBUG_PREVIEW]:
    os.makedirs(folder, exist_ok=True)

# ---- Load trained model ----
with open(MODEL_PATH, "rb") as f:
    knn = pickle.load(f)


def extract_histogram(image_path, bins=HIST_BINS):
    img = cv2.imread(str(image_path))
    if img is None:
        raise Exception(f"Could not read: {image_path}")
    img_small = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten(), img_small


def classify_and_copy(img_path):
    try:
        feature, img_small = extract_histogram(img_path)
        pred = knn.predict([feature])[0]
        label = "full_tank" if pred == 1 else "detail_shot"
        dest = DEST_FULL if pred == 1 else DEST_DETAIL

        # Copy file
        shutil.copy2(img_path, os.path.join(dest, os.path.basename(img_path)))
        print(f"[{label.upper()}] {os.path.basename(img_path)}")

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")


def main():
    # List available years/folders
    subfolders = sorted(
        [f for f in os.listdir(ARCHIVE_ROOT) if (Path(ARCHIVE_ROOT) / f).is_dir()]
    )
    print("Available years/folders:")
    for idx, name in enumerate(subfolders):
        print(f"{idx+1}: {name}")
    print()

    # Pick year/folder
    while True:
        try:
            choice = int(
                input(f"Pick a year/folder to classify [1-{len(subfolders)}]: ")
            )
            if 1 <= choice <= len(subfolders):
                chosen_folder = subfolders[choice - 1]
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    src_dir = Path(ARCHIVE_ROOT) / chosen_folder
    images = list(src_dir.glob("*.jpg"))
    print(f"\nClassifying {len(images)} images in {src_dir}...\n")

    for img_path in images:
        classify_and_copy(img_path)

    print("\n==== DONE ====")
    print(
        f"Results in:\n- {DEST_FULL}\n- {DEST_DETAIL}\nDebug images in:\n- {DEBUG_PREVIEW}"
    )


if __name__ == "__main__":
    main()
