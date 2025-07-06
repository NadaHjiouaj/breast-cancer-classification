
# 🩺 Breast Cancer Classification from Ultrasound Images

**Deep Learning model for classifying breast tumors (Benign, Malignant, Normal) using ResNet-18**

##  Project Overview

This project implements an end-to-end deep learning pipeline to classify **breast ultrasound images** into three categories:

* 🟢 **Benign** — 437 images
* 🔴 **Malignant** — 210 images
* 🔷 **Normal** — 133 images

It uses **transfer learning** with a modified ResNet-18 architecture and advanced data augmentation techniques to improve generalization.

---

## 🚀 Features

* ✅ End-to-end pipeline: from exploratory data analysis to model evaluation
* ✅ Transfer Learning with frozen ResNet-18 base and custom classifier
* ✅ Advanced data augmentation (including Mixup, flips, affine transforms)
* ✅ Reproducible experiments (random seeds set)
* ✅ Performance visualizations (accuracy, F1-score, loss)

---

## 📂 Dataset

We use the [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), which contains labeled ultrasound images of breast tumors.

### Data Structure

```
project/
├── data/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── src/
    └── breast-cancer-classification.ipynb
```

> ⚠️ **Note:** You need to manually download the dataset from Kaggle and place it under the `data/` folder.

---

## 🛠️ Technical Details

### Model Architecture

We use **ResNet-18** as backbone with the following custom classifier head:

```python
ResNet18(
  (frozen_layers): [...]
  (fc): Sequential(
    Dropout(p=0.85),
    Linear(512 → 128),
    BatchNorm1d(128),
    ReLU(),
    Dropout(p=0.5),
    Linear(128 → 3)
  )
)
```

### Data Augmentation

Applied on training images:

* Resize: `256×256`
* Horizontal flip: `p=0.3`
* Vertical flip: `p=0.3`
* Affine transform: `rotation ±10°, translation 10%`
* Color jitter: `brightness & contrast ±10%`
* Random erasing: `p=0.3`
* Normalization:

  ```
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
  ```

---

### Performance Metrics

| Metric   | Training | Validation | Test  |
| -------- | -------- | ---------- | ----- |
| Accuracy | 94.2%    | 91.8%      | 92.4% |
| F1-Score | 0.938    | 0.912      | 0.915 |
| Loss     | 0.183    | 0.221      | 0.214 |

---

## 💻 How to Run

### 📋 Prerequisites

* Python ≥ 3.8
* PyTorch ≥ 2.0
* GPU recommended for training

### 📦 Install dependencies

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
```

Or open directly in **Google Colab** using the badge above.

---

