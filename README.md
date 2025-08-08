# SavannahTankVision
Built with CNNs, kNN, and traditional ML to classify thousands of industrial tank images into full vs. detail shots. Streamlines photo archiving and sets the stage for searchable image databases.
# SavannahTankVision

**Classify. Archive. Organize.**  
Built for real-world speed, SavannahTankVision uses CNNs, kNN, and traditional ML to automatically classify thousands of industrial tank photos into full-tank or detail shots — cutting manual cleanup time and enabling searchable archives for engineering, QA, and marketing use cases.

---

## 🔧 Tech Stack

- **Python** (OpenCV, NumPy, scikit-learn, PyTorch)
- **Machine Learning**:  
  - `k-Nearest Neighbors` for baseline classification  
  - `Convolutional Neural Networks (CNN)` for advanced accuracy
- **Computer Vision**: Color histograms, HSV space
- **Tools**: Streamlit-ready, P-drive integration, Visual Studio Code, Windows/Remote-friendly

---

## 🧠 What It Does

- Loads labeled training images (full tanks vs. detail shots)
- Extracts image features (color histograms or CNN embeddings)
- Trains a classifier (kNN or CNN-based)
- Scans tank job folders on Savannah Tank's network drive
- Automatically sorts photos into proper folders:
  - `full_tanks/`
  - `detail_shots/`

Optional debug mode available for misclassified previews during tuning.

---

## 📁 Project Structure

TankPhotoArchive/
├── training_sets/
│ ├── full_tanks_train/
│ └── detail_shots_train/
├── classified/
│ ├── full_tanks/
│ └── detail_shots/
├── TankPhotoArchive_Scripts/
│ ├── classify_tank_photos_knn.py
│ ├── classify_tank_photos_ai.py
│ ├── AI_Training_Script.py
│ └── requirements_CV.txt


---

## 🚀 How To Use

### 🧠 Train the Model
1. Place labeled images in `training_sets/full_tanks_train/` and `training_sets/detail_shots_train/`.
2. Run `AI_Training_Script.py` to train your CNN or `train_knn.py` for a lightweight model.
3. A `.pkl` model file will be saved for later use.

### 📷 Classify Images
1. Run `classify_tank_photos_ai.py` or `classify_tank_photos_knn.py`.
2. Select the year/folder to process.
3. Model will classify and move images into `classified/full_tanks/` or `classified/detail_shots/`.

---

## ✅ Why It Matters

This app was built for Savannah Tank & Equipment Corp to streamline internal image organization across tens of thousands of archived tank photos. What used to be a tedious manual task can now be done in minutes — freeing up engineering and marketing time and laying the groundwork for future image search and content reuse.

---

## 🧠 Future Plans

- Add Streamlit interface for ease of use
- Tag detection (e.g., cone bottom, half-pipe, ladder)
- Connect to searchable metadata/image database

---

## 👩‍💻 Author

Kelly Arseneau  
[Portfolio](https://sites.google.com/view/kelly-ds-portfolio?usp=sharing) | [GitHub](https://github.com/kelly12201984) | [LinkedIn](https://www.linkedin.com/in/kelly-arseneau-9459b1273/)

---

> “Don’t organize your chaos — eliminate it.”  
> — SavannahTankVision
