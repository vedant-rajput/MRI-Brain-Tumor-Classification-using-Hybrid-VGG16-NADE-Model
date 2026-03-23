# MRI-Brain-Tumor-Classification-using-Hybrid-VGG16-NADE-Model

A deep learning pipeline for classifying brain MRI scans into four tumour categories using a hybrid architecture that combines VGG16 (transfer
learning backbone) with a NADE (Neural Autoregressive Distribution Estimator) module.

Architechture
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

Why NADE?
NADE models the joint distribution of features autoregressively:
p(x) = ∏_d  p(x_d | x_{1:d-1})
Applied after VGG16's feature extraction, it:

Learns the density of the feature distribution, acting as a
learned regulariser on top of the CNN representations.
Provides an auxiliary density loss that combats overfitting on
the relatively small medical imaging dataset.
Improves generalisation on the minority classes (meningioma,
notumor).

Brain Tumor MRI Dataset — Kaggle
masoudnickparvar/brain-tumor-mri-dataset

glioma --> Malignant tumour from glial cells
meningioma --> Usually benign, arises from meninges
notumor --> Healthy brain scan
pituitary --> Tumour on the pituitary gland

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
