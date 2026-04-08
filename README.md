# 🦟 Malaria Detection with Deep Learning

A binary image classification system to automatically detect malaria-infected (parasitized) blood cells from microscopic images, using Convolutional Neural Networks (CNNs) and Transfer Learning.

---

## 📋 Problem Statement

Manual microscopy for malaria diagnosis is time-intensive and requires specialized expertise often unavailable in rural or resource-limited settings. In 2022, the WHO recorded over **249 million malaria cases** and **608,000 deaths** across 85 countries, with 80% of deaths in the African region being children under 5.

This project proposes an automated detection system that processes blood smear images to distinguish **Parasitized** cells (containing *Plasmodium* parasite) from **Uninfected** cells — delivering fast, accurate diagnoses without manual interpretation.

---

## 📁 Dataset

- **Source:** Microscopic blood smear images (provided as a zip archive)
- **Total images:** 27,303 (colored)
  - Train set: **24,958 images** (~49.6% uninfected / 50.4% parasitized — nearly balanced)
  - Test set: **2,345 images** (~55.4% uninfected / 44.6% parasitized)
- **Classes:** `Parasitized` (label=1) / `Uninfected` (label=0)
- **Image size:** Variable → resized to **64×64** pixels for modeling

---

## 🎯 Objectives

- Load, label, and explore the image dataset
- Apply preprocessing techniques: RGB→HSV conversion, Gaussian blurring
- Perform data augmentation to enhance training diversity
- Build and compare multiple CNN architectures
- Improve performance via Transfer Learning (VGG16)

---

## 🔬 Methodology

### Preprocessing
- Images resized to 64×64 and converted to NumPy 4D arrays
- Pixel values normalized to [0, 1]
- **RGB → HSV conversion** to highlight hue/saturation differences between classes
- **Gaussian blurring** (kernel 5×5) to study broader cell shape patterns

### Data Augmentation
Applied via `ImageDataGenerator`:
- Random rotation (±25°)
- Shear and zoom (up to 0.2–0.3)
- Horizontal flipping
- Nearest-fill mode

---

## 🏗️ Models

| Model | Architecture | Key Features | Test Accuracy |
|-------|-------------|--------------|---------------|
| **Base Model** | 2 Conv + 2 Dense | ReLU, Dropout | ~98.5% |
| **Model 1** | 3 Conv + 2 Dense | ReLU, Dropout | Improved |
| **Model 2** | 3 Conv + 2 Dense | BatchNorm, LeakyReLU | ~97.9% |
| **Model 3** | 3 Conv + 2 Dense | BatchNorm + Data Augmentation | — |
| **VGG16** | Pretrained (frozen) + custom head | Transfer Learning (ImageNet) | ~94% |
| **HSV Model** | 3 Conv + 2 Dense | BatchNorm, LeakyReLU on HSV images | — |

All models use:
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary cross-entropy
- **Callbacks:** EarlyStopping + ModelCheckpoint
- **Evaluation:** Accuracy, classification report, confusion matrix

> ⚠️ **Note on metric choice:** In a healthcare context, **recall** is prioritized over accuracy — a false negative (missed parasite) is more dangerous than a false positive.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Image processing | `OpenCV`, `Pillow`, `NumPy` |
| Visualization | `Matplotlib`, `Seaborn` |
| Modeling | `TensorFlow / Keras` |
| Evaluation | `scikit-learn` |
| Environment | Google Colab + Google Drive |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/malaria-detection.git
cd malaria-detection
```

### 2. Install dependencies
```bash
pip install tensorflow numpy opencv-python pillow matplotlib seaborn scikit-learn
```

### 3. Prepare the dataset
Download the `cell_images.zip` dataset and upload it to your Google Drive at:
```
/MyDrive/Learning/ADSP/Capstone_Project/cell_images.zip
```
Then unzip into:
```
/MyDrive/Learning/ADSP/Capstone_Project/datasets/
```

### 4. Run the notebook
Open `Project_Malaria_Detection_Full_Code.ipynb` in Google Colab and run all cells.

---

## 📊 Key Results & Insights

- The **Base CNN** achieves ~98.5% test accuracy, with slightly better performance on the uninfected class (F1: 0.99 vs 0.98 for parasitized).
- **Batch Normalization + LeakyReLU** (Model 2) stabilizes training but reduces recall on parasitized cells to 0.77 — not ideal for clinical use.
- **Data augmentation** improves generalization by diversifying the training set with realistic transformations.
- **VGG16 Transfer Learning** reaches ~94% accuracy with only 10 epochs and 566K trainable parameters, showing strong potential with more training time.
- **HSV color space** highlights parasite-related hue/saturation features, making it a promising input alternative for future exploration.
- Gaussian blurring reduces noise but also eliminates fine-grained parasite details — less suited to this specific task.

---

## 🔮 Future Improvements

- Train VGG16 (or other architectures like ResNet50, EfficientNet) for more epochs with fine-tuning
- Explore deeper custom CNN architectures
- Optimize recall for the parasitized class specifically (threshold tuning, class-weighted loss)
- Deploy the best model as a REST API (FastAPI + Docker)
- Integrate Grad-CAM for visual explainability

---

## 📌 Project Structure

```
malaria-detection/
│
├── Project_Malaria_Detection_Full_Code.ipynb   # Main notebook
├── README.md                                    # This file
├── best_model_weights.keras                     # Saved weights (Model 3 w/ augmentation)
├── vgg_model_weights.keras                      # Saved weights (VGG16)
└── best_hsv_model_weights.keras                 # Saved weights (HSV model)
```

---

## 📄 License

This project is for educational purposes as part of the MIT Applied Data Science Program capstone.
