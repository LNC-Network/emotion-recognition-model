# 🧠 Model Trainer: CNN-Based Image Classification

A simple and extensible Python project to train CNN on a dataset of grayscale (black-and-white) and RGB images using PyTorch. 

---

## ⚙️ Setup

> Recommended: Use a virtual environment to isolate dependencies.

### 1. Create a virtual environment

```bash
python -m venv .venv
````

### 2. Activate the virtual environment

* On **Windows**:

```bash
.venv\Scripts\activate
```

* On **macOS/Linux**:

```bash
source .venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

## 📦 Downloading the Dataset

We used a Kaggle dataset that contains a combination of grayscale and RGB images. 

It should be organized as follows after extraction:

```
data/
├── train/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```


## 🧠 Model Notes

* A **CNN (Convolutional Neural Network)** is used for significantly better performance on image data.
* RGB and grayscale images are normalized and processed to be compatible with the CNN input.

---

## 📌 Requirements

Basic dependencies (defined in `requirements.txt`) include:

* `torch`
* `torchvision`
* `numpy`
* `matplotlib`
* `opencv-python`
* `scikit-learn`

---

