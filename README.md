# MRI Brain Tumor Classification — Hybrid VGG16-NADE Model

A deep learning pipeline for classifying brain MRI scans into four tumour
categories using a hybrid architecture that combines **VGG16** (transfer
learning backbone) with a **NADE** (Neural Autoregressive Distribution
Estimator) module.

---

## Architecture

```
Input (224×224×3)
      │
 ┌────▼──────────────────────────────────┐
 │  VGG16 Backbone  (ImageNet weights)   │
 │  Frozen (phase 1) → fine-tuned block5 │
 └────┬──────────────────────────────────┘
      │  Global Average Pooling  (512,)
      │
 ┌────▼──────────────────────────────────────────────┐
 │  Feature Projection  →  NADE_INPUT_DIM (512,)     │
 │  L2-normalised                                    │
 └────┬──────────────────────────────────────────────┘
      │
 ┌────▼────────────────────────────────────────────────────┐
 │  NADE Module                                            │
 │  ─ K autoregressive masked passes over feature vector   │
 │  ─ Produces density-regularised representation          │
 │  ─ Adds auxiliary BCE density loss to total loss        │
 └────┬────────────────────────────────────────────────────┘
      │  Concatenate [projection ‖ NADE repr]
      │
 ┌────▼────────────────────────────────────┐
 │  Classification Head                    │
 │  Dense(256) → BN → Dropout(0.4)         │
 │  Dense(128) → BN → Dropout(0.3)         │
 │  Dense(4, softmax)                      │
 └─────────────────────────────────────────┘
```

### Why NADE?

NADE models the joint distribution of features autoregressively:

```
p(x) = ∏_d  p(x_d | x_{1:d-1})
```

Applied after VGG16's feature extraction, it:
1. Learns the **density** of the feature distribution, acting as a
   learned regulariser on top of the CNN representations.
2. Provides an auxiliary **density loss** that combats overfitting on
   the relatively small medical imaging dataset.
3. Improves generalisation on the minority classes (`meningioma`,
   `notumor`).

---

## Dataset

**Brain Tumor MRI Dataset** — Kaggle  
`masoudnickparvar/brain-tumor-mri-dataset`

| Class | Description |
|---|---|
| `glioma` | Malignant tumour from glial cells |
| `meningioma` | Usually benign, arises from meninges |
| `notumor` | Healthy brain scan |
| `pituitary` | Tumour on the pituitary gland |

~7,000 training images · ~1,300 test images

---

## Project Structure

```
mri_tumor_classifier/
├── config.py                   # All hyperparameters & paths
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Generators, tf.data pipeline, augmentation
│   ├── model.py                # VGG16-NADE architecture (NADELayer, FeatureProjection)
│   ├── train.py                # Two-phase training pipeline + CLI
│   ├── evaluate.py             # Metrics, plots, Grad-CAM
│   ├── predict.py              # Single/batch inference + MC-Dropout uncertainty
│   └── utils.py                # Seeding, GPU setup, Kaggle download
│
├── notebooks/
│   └── MRI_Brain_Tumor_VGG16_NADE.ipynb   # End-to-end walkthrough
│
├── data/
│   ├── raw/                    # Kaggle dataset (Training/ + Testing/)
│   └── processed/              # Reserved for future preprocessing artefacts
│
├── models/
│   └── saved/                  # Keras .keras model checkpoints
│
└── results/
    ├── plots/                  # Confusion matrix, ROC curves, Grad-CAM gallery
    └── reports/                # Classification report JSON, training CSVs
```
My first formal exposure to research came at JNTUH, where I worked under faculty supervision in the Department of Computer Science & Engineering on a hybrid deep learning architecture for MRI brain tumour classification. The core research question fascinated me: could generative distribution modelling, traditionally used for density estimation, be fused with a discriminative CNN feature extractor to outperform standard computer vision architectures in low-data medical imaging settings? I designed and executed the experimental pipeline combining VGG16 transfer learning with Neural Autoregressive Distribution Estimation (NADE), ultimately achieving 97% multi-class classification accuracy a 7-percentage-point improvement over the VGG16 baseline.

Beyond the model itself, I learned what rigorous research actually demands: extensive literature review, reproducible training pipelines, careful data augmentation strategy, and writing that survives peer review. The work was published at the 3rd International Conference on Innovations and Recent Trends in Computer Science (ICIRTCS-23-064) in December 2023, marking my first peer-reviewed publication. This experience shaped how I approach every project since domain-aware, hypothesis-driven, and obsessed with making results reproducible. It's the reason my later work on NASA geomagnetic forecasting leaned on physics-informed features rather than raw inputs, and why I treat experimental rigor as non-negotiable.

Since, vgg16_nade_best.keras is 127.99 MB (Model) and GitHub has a 100 MB file size limit. Make sure to download it from the link below, and then past the complete folder according to the project architechture.

If your only able to download the (vgg16_nade_best.keras), after downloading it
Follow these steps:

--> Create a new folder on your desktop called as models
--> indside the models folder create another folder named 'saved'
--> then paste the downloaded file vgg16_nade_best.keras

https://drive.google.com/drive/folders/1MjVva21p1Sa9p_S19gtK298Yt8Xh0P7b?usp=sharing

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
# Place your kaggle.json in ~/.kaggle/ first
python -c "from src.utils import download_dataset; download_dataset()"
```

Or download manually from Kaggle and unzip into `data/raw/` so the
structure is `data/raw/Training/<class>/` and `data/raw/Testing/<class>/`.

### 3. Train

```bash
# Two-phase training (phase 1: frozen, phase 2: fine-tune)
python -m src.train

# Custom epochs
python -m src.train --epochs_p1 15 --epochs_p2 25 --batch_size 16
```

### 4. Evaluate

```bash
python -m src.evaluate
```

Outputs saved to `results/plots/` and `results/reports/`.

### 5. Predict

```bash
# Single image
python -m src.predict --image data/raw/Testing/glioma/Te-gl_0010.jpg --save_plot

# With Monte Carlo Dropout uncertainty (30 forward passes)
python -m src.predict --image path/to/mri.jpg --mc_samples 30

# Batch folder
python -m src.predict --folder data/raw/Testing/meningioma/
```

### 6. Notebook

```bash
cd notebooks
jupyter notebook MRI_Brain_Tumor_VGG16_NADE.ipynb
```

---

## Training Strategy

| Phase | VGG16 | Epochs | LR |
|---|---|---|---|
| 1 — Head training | Frozen | 10 | 1e-4 |
| 2 — Fine-tuning | block5 trainable | 20 | 1e-5 |

### Callbacks (both phases)
- `ModelCheckpoint` — saves best `val_accuracy`
- `ReduceLROnPlateau` — halves LR on val_loss plateau (patience=4)
- `EarlyStopping` — restores best weights (patience=8)
- `CSVLogger` — full epoch metrics log
- `TensorBoard` — histogram logging

---

## Key Features

| Feature | Implementation |
|---|---|
| Transfer learning | VGG16 ImageNet weights |
| NADE density reg. | Custom `NADELayer` Keras layer |
| Imbalance handling | Sklearn balanced class weights |
| Augmentation | Rotation, shift, zoom, flip, brightness |
| Uncertainty | Monte Carlo Dropout (inference) |
| Explainability | Grad-CAM on `block5_conv3` |
| Metrics | Accuracy, AUC, Precision, Recall, F1 |

---

## Expected Results

On the standard Brain Tumor MRI Dataset:

| Metric | Target |
|---|---|
| Test Accuracy | ~97–98% |
| Macro F1 | ~0.97 |
| AUC (mean OvR) | ~0.999 |

---

## References

- Simonyan & Zisserman (2015). *Very Deep Convolutional Networks for
  Large-Scale Image Recognition*. ICLR 2015.
- Larochelle & Murray (2011). *The Neural Autoregressive Distribution
  Estimator*. AISTATS 2011.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localisation*. ICCV 2017.
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*.
  ICML 2016.
