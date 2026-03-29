"""
predict.py
──────────
Single-image and batch-folder inference with uncertainty estimation.

Usage
─────
  # Single image
  python -m src.predict --image path/to/mri.jpg

  # Folder
  python -m src.predict --folder path/to/mris/

  # With Monte Carlo Dropout uncertainty
  python -m src.predict --image path/to/mri.jpg --mc_samples 30
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import preprocess_single_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════

_MODEL_CACHE: keras.Model = None

def get_model(model_path: str = config.BEST_MODEL_PATH) -> keras.Model:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from src.model import NADELayer, FeatureProjection
        _MODEL_CACHE = keras.models.load_model(
            model_path,
            custom_objects={"NADELayer": NADELayer,
                            "FeatureProjection": FeatureProjection},
        )
        logger.info(f"Model loaded from {model_path}")
    return _MODEL_CACHE


# ══════════════════════════════════════════════════════════════
#  Single-image prediction
# ══════════════════════════════════════════════════════════════

def predict_single(
    image_path: str,
    model_path: str = config.BEST_MODEL_PATH,
    mc_samples: int = 0,
) -> dict:
    """
    Predict tumour class for a single MRI image.

    Parameters
    ──────────
    image_path  : path to the MRI image
    mc_samples  : if > 0, run MC-Dropout with this many forward passes
                  to estimate epistemic uncertainty

    Returns
    ───────
    dict with keys:
        predicted_class  : str
        confidence       : float (0-1)
        probabilities    : dict {class_name: probability}
        uncertainty      : float | None  (std of MC predictions)
    """
    model    = get_model(model_path)
    img      = preprocess_single_image(image_path)

    if mc_samples > 0:
        # Enable dropout at inference (training=True)
        preds = np.stack([
            model(img, training=True).numpy()
            for _ in range(mc_samples)
        ])                              # (mc_samples, 1, num_classes)
        mean_pred   = preds[:, 0, :].mean(axis=0)
        uncertainty = preds[:, 0, :].std(axis=0).mean()
    else:
        mean_pred   = model.predict(img, verbose=0)[0]
        uncertainty = None

    pred_idx   = int(mean_pred.argmax())
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = float(mean_pred[pred_idx])

    return {
        "predicted_class": pred_class,
        "confidence":      round(confidence, 4),
        "probabilities":   {c: round(float(p), 4)
                            for c, p in zip(config.CLASS_NAMES, mean_pred)},
        "uncertainty":     round(float(uncertainty), 4) if uncertainty else None,
    }


# ══════════════════════════════════════════════════════════════
#  Batch-folder prediction
# ══════════════════════════════════════════════════════════════

def predict_folder(
    folder: str,
    model_path: str = config.BEST_MODEL_PATH,
    mc_samples: int = 0,
) -> list[dict]:
    """Run predict_single on every image in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)
    results = []
    for p in paths:
        try:
            res = predict_single(str(p), model_path, mc_samples)
            res["image"] = str(p)
            results.append(res)
            logger.info(f"{p.name:<40}  →  {res['predicted_class']} ({res['confidence']:.2%})")
        except Exception as e:
            logger.warning(f"Skipped {p.name}: {e}")
    return results


# ══════════════════════════════════════════════════════════════
#  Probability bar chart for a single prediction
# ══════════════════════════════════════════════════════════════

def visualise_prediction(
    result: dict,
    image_path: str,
    save_path: str = None,
):
    """Save a two-panel figure: original MRI + probability bars."""
    import cv2

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))

    probs  = [result["probabilities"][c] for c in config.CLASS_NAMES]
    colors = ["#55A868" if c == result["predicted_class"] else "#4C72B0"
              for c in config.CLASS_NAMES]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    ax_img.imshow(img, cmap="gray")
    ax_img.axis("off")
    ax_img.set_title(
        f"Prediction: {result['predicted_class'].upper()}\n"
        f"Confidence: {result['confidence']:.2%}"
        + (f"\nUncertainty (σ): {result['uncertainty']:.4f}"
           if result["uncertainty"] else ""),
        fontsize=11, fontweight="bold",
    )

    bars = ax_bar.barh(config.CLASS_NAMES, probs, color=colors, edgecolor="white")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probability", fontsize=10)
    ax_bar.set_title("Class Probabilities", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)
    for bar, p in zip(bars, probs):
        ax_bar.text(min(p + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                    f"{p:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "prediction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Prediction figure → {save_path}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG16-NADE inference")
    parser.add_argument("--image",      type=str, help="Path to a single MRI image")
    parser.add_argument("--folder",     type=str, help="Path to a folder of MRI images")
    parser.add_argument("--model_path", type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument("--mc_samples", type=int, default=0,
                        help="MC-Dropout forward passes (0 = disabled)")
    parser.add_argument("--save_plot",  action="store_true",
                        help="Save a prediction visualisation")
    args = parser.parse_args()

    if args.image:
        result = predict_single(args.image, args.model_path, args.mc_samples)
        print("\n── Prediction ──────────────────────────────")
        for k, v in result.items():
            print(f"  {k:<20}: {v}")
        if args.save_plot:
            visualise_prediction(result, args.image)

    elif args.folder:
        results = predict_folder(args.folder, args.model_path, args.mc_samples)
        print(f"\n  Processed {len(results)} images.")
    else:
        parser.print_help()
