"""
Configuration file for MRI Brain Tumor Classification
Hybrid VGG16-NADE Model
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR     = os.path.join(RESULTS_DIR, "reports")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
# Kaggle dataset: "masoudnickparvar/brain-tumor-mri-dataset"
# Classes: glioma | meningioma | notumor | pituitary
CLASS_NAMES     = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES     = len(CLASS_NAMES)

TRAIN_DIR       = os.path.join(RAW_DATA_DIR, "Training")
TEST_DIR        = os.path.join(RAW_DATA_DIR, "Testing")

# ─────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
IMG_CHANNELS    = 3
INPUT_SHAPE     = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE          = 32
EPOCHS_FROZEN       = 10    # train only custom head (VGG16 frozen)
EPOCHS_FINETUNE     = 20    # fine-tune last VGG16 block + head
LEARNING_RATE       = 1e-4
FINETUNE_LR         = 1e-5
VALIDATION_SPLIT    = 0.15
RANDOM_SEED         = 42

# ─────────────────────────────────────────────
# NADE SETTINGS
# ─────────────────────────────────────────────
NADE_HIDDEN_UNITS   = 256   # hidden units per NADE mask step
NADE_INPUT_DIM      = 512   # feature dimension fed into NADE
NADE_STEPS          = 4     # autoregressive factorisation steps

# ─────────────────────────────────────────────
# VGG16 SETTINGS
# ─────────────────────────────────────────────
VGG16_UNFREEZE_FROM = "block5_conv1"   # layer name to start fine-tuning

# ─────────────────────────────────────────────
# MODEL CHECKPOINT
# ─────────────────────────────────────────────
BEST_MODEL_PATH     = os.path.join(MODEL_DIR, "vgg16_nade_best.keras")
FINAL_MODEL_PATH    = os.path.join(MODEL_DIR, "vgg16_nade_final.keras")
