
# Butterfly & Moth Species Classification

![Project Banner](images/banner.png)  <!-- Optional: add a relevant image -->

## Project Overview

This project focuses on **automatically classifying butterfly and moth species** using deep learning and transfer learning techniques. The goal is to build a robust machine learning pipeline that can identify **100 different species** from images, helping enthusiasts, researchers, and conservationists in species recognition.

Key highlights:
- Dataset: ~13,500 labeled images across 100 species
- Deep learning model: Transfer learning with **EfficientNetB4** backbone
- Preprocessing & augmentation: Resize, normalization, flips, rotations, brightness adjustments
- Achieved reliable predictions with visualizations for single and multiple images

---

## Project Structure

butterfly-classification/
├── Butterfly/
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/             # colab notebooks for EDA & training
├── label_map.json.     # mapping of numeric labels to species names
├── best_model.keras       # Saved trained model
├── requirements.txt.         #  dependencies list
└── README.md


## Dataset

- The dataset consists of butterfly and moth images organized in:
```
butterfly/
├── train/
├── valid/
└── test/
- Each class has its own subfolder (species name).
- Total: **~13,500 images** (train: 12,594, validation: 500, test: 500)
- Source: Publicly available butterfly and moth images ([Kaggle / Open datasets link can be added])
```
---

## Installation

Clone this repository and install dependencies:

```bash
git clone <your-repo-url>
cd butterfly-classification
pip install -r requirements.txt

##  Key Dependencies:
* Python >= 3.9
* TensorFlow / Keras
* Pandas, NumPy, Matplotlib
* Scikit-learn

##  Usage
1. Prepare Dataset
* Organize images into train/, valid/, and test/ directories.
* Optional: check class balance, preprocess images, resize to 224x224 pixels.
2. Train the Model
* Transfer learning with EfficientNetB4 backbone
* Freeze base layers initially
* Train on full dataset with class weights to handle imbalance
* Fine-tune top layers for improved performance

# Example: Load model and train
from tensorflow.keras.models import load_model
model = load_model("best_model.keras")

3. Make Predictions

Single Image

from predict import predict_single_image

img_path = "test/ADONIS/001.jpg"
label, conf, img = predict_single_image(img_path)

print(f"Predicted species: {label} ({conf:.1%})")

Multiple Images

from predict import predict_multiple_images

img_paths = ["test/ADONIS/001.jpg", "test/ZEBRA LONG WING/024.jpg"]
labels, confs, images = predict_multiple_images(img_paths)

4. Visualize Predictions
* Display images with predicted species and confidence
* Optional: plot top 5 predicted classes for each image

## Model Evaluation
* Metrics:
    * Accuracy
    * Precision, Recall, F1-score (macro-averaged)
* Confusion matrix for top classes to analyze performance
* Normalized confusion matrix for readability

##  Deployment
* The trained model is saved as best_model.keras
* Label ↔ species mapping saved as a dictionary for inference
* Deployed with: Gradio for interactive demos

##  References
* TensorFlow Transfer Learning Guide
* EfficientNet Paper: Tan and Le, 2019, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”
* Public Butterfly/Moth datasets (add source links)

##  License
This project is licensed under the MIT License.







