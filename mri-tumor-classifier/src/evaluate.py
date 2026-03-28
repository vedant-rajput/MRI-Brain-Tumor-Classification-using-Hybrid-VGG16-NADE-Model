"""
evaluate.py
───────────
Full evaluation suite for the trained VGG16-NADE model:
  • Per-class precision / recall / F1
  • Confusion matrix (normalised + raw)
  • ROC-AUC curves (one-vs-rest)
  • Grad-CAM visualisations
  • Training history curves
  • Misclassification gallery
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import build_generators, preprocess_single_image
from src.model import build_vgg16_nade_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(config.PLOTS_DIR,   exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ══════════════════════════════════════════════════════════════
#  Load model
# ══════════════════════════════════════════════════════════════

def load_model(model_path: str = config.BEST_MODEL_PATH) -> keras.Model:
    from src.model import NADELayer, FeatureProjection
    model = keras.models.load_model(
        model_path,
        custom_objects={"NADELayer": NADELayer, "FeatureProjection": FeatureProjection},
    )
    logger.info(f"Loaded model from {model_path}")
    return model


# ══════════════════════════════════════════════════════════════
#  Confusion Matrix
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, save_dir: str = config.PLOTS_DIR):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(title,        fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix → {save_path}")


# ══════════════════════════════════════════════════════════════
#  ROC-AUC
# ══════════════════════════════════════════════════════════════

def plot_roc_curves(y_true_bin, y_proba, save_dir: str = config.PLOTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(config.CLASS_NAMES, PALETTE)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — One-vs-Rest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    save_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Training History
# ══════════════════════════════════════════════════════════════

def plot_training_history(history_path: str = None, save_dir: str = config.PLOTS_DIR):
    if history_path is None:
        history_path = os.path.join(config.REPORTS_DIR, "combined_history.json")
    if not os.path.exists(history_path):
        logger.warning("No combined_history.json found — skipping history plot.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("accuracy",  "val_accuracy",  "Accuracy"),
        ("loss",      "val_loss",      "Loss"),
        ("auc",       "val_auc",       "AUC"),
    ]

    for ax, (train_key, val_key, title) in zip(axes, metrics):
        if train_key not in history:
            continue
        ax.plot(epochs, history[train_key], label="Train", color="#4C72B0", lw=2)
        if val_key in history:
            ax.plot(epochs, history[val_key], label="Validation", color="#DD8452", lw=2, ls="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("VGG16-NADE Training History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Grad-CAM
# ══════════════════════════════════════════════════════════════

def grad_cam(model: keras.Model, img_array: np.ndarray,
             last_conv_layer: str = "block5_conv3") -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a preprocessed image array (1,H,W,3).
    Returns heatmap of shape (H, W) normalised to [0, 1].
    """
    # Sub-model that outputs the conv layer + final predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("vgg16").get_layer(last_conv_layer).output,
                 model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_class = tf.argmax(predictions[0])
        loss       = predictions[:, pred_class]

    grads       = tape.gradient(loss, conv_outputs)              # (1,7,7,512)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))         # (512,)

    conv_outputs = conv_outputs[0]                               # (7,7,512)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]  # (7,7,1)
    heatmap      = tf.squeeze(heatmap).numpy()
    heatmap      = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def plot_grad_cam_gallery(
    model: keras.Model,
    test_gen,
    n_samples: int = 8,
    save_dir: str = config.PLOTS_DIR,
):
    """
    Plot a gallery of Grad-CAM overlays for the first n_samples test images.
    """
    import cv2

    batch_imgs, batch_labels = next(test_gen)
    n = min(n_samples, len(batch_imgs))

    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))

    for i in range(n):
        img_arr  = batch_imgs[i:i+1]          # (1,224,224,3)
        true_cls = config.CLASS_NAMES[np.argmax(batch_labels[i])]
        pred_cls = config.CLASS_NAMES[np.argmax(model.predict(img_arr, verbose=0))]

        # Original image (undo VGG preprocess: add mean back)
        orig = img_arr[0].copy()
        orig[..., 0] += 103.939
        orig[..., 1] += 116.779
        orig[..., 2] += 123.68
        orig = np.clip(orig, 0, 255).astype(np.uint8)
        orig = orig[..., ::-1]   # BGR → RGB

        # Grad-CAM heatmap
        heatmap = grad_cam(model, img_arr)
        heatmap_resized = cv2.resize(heatmap, (config.IMG_WIDTH, config.IMG_HEIGHT))
        heatmap_color   = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )[..., ::-1]
        overlay = cv2.addWeighted(orig, 0.55, heatmap_color, 0.45, 0)

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        color = "green" if true_cls == pred_cls else "red"
        axes[0, i].set_title(f"True: {true_cls}\nPred: {pred_cls}",
                              fontsize=7.5, color=color)

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        axes[1, i].set_title("Grad-CAM", fontsize=7.5)

    plt.suptitle("Grad-CAM Explanations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "grad_cam_gallery.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM gallery → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Per-class bar chart
# ══════════════════════════════════════════════════════════════

def plot_per_class_metrics(report_dict: dict, save_dir: str = config.PLOTS_DIR):
    metrics = ["precision", "recall", "f1-score"]
    x       = np.arange(config.NUM_CLASSES)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (metric, color) in enumerate(zip(metrics, ["#4C72B0", "#55A868", "#DD8452"])):
        values = [report_dict[cls][metric] for cls in config.CLASS_NAMES]
        ax.bar(x + j * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(config.CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class metrics → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Master evaluate function
# ══════════════════════════════════════════════════════════════

def evaluate(model_path: str = config.BEST_MODEL_PATH):
    model = load_model(model_path)

    _, _, test_gen = build_generators()

    logger.info("Running predictions on test set …")
    y_proba = model.predict(test_gen, verbose=1)
    y_pred  = y_proba.argmax(axis=1)
    y_true  = test_gen.classes
    y_true_bin = keras.utils.to_categorical(y_true, config.NUM_CLASSES)

    # ── Text report ───────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True
    )
    report_str = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_str)

    report_path = os.path.join(config.REPORTS_DIR, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true_bin, y_proba)
    plot_training_history()
    plot_per_class_metrics(report)
    plot_grad_cam_gallery(model, test_gen)

    # ── Summary metric ────────────────────────────────────────
    acc      = (y_pred == y_true).mean()
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"\n  Overall Accuracy : {acc:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")

    return report


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=config.BEST_MODEL_PATH)
    args = parser.parse_args()
    evaluate(args.model_path)
