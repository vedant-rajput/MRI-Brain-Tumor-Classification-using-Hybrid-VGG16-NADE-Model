"""
train.py
────────
Two-phase training pipeline for the VGG16-NADE hybrid model.

Phase 1  – VGG16 backbone frozen, train classification head + NADE
Phase 2  – Unfreeze VGG16 block5, fine-tune end-to-end at low LR
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import build_generators, compute_class_weights
from src.model import (
    build_vgg16_nade_model,
    compile_phase1,
    unfreeze_for_finetuning,
    print_model_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Ensure output directories exist
# ─────────────────────────────────────────────────────────────
for _d in (config.MODEL_DIR, config.PLOTS_DIR, config.REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════

def make_callbacks(phase: int) -> list:
    """Build a standard callback set for a given training phase."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    callbacks = [
        # Save the best checkpoint by val_accuracy
        keras.callbacks.ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Reduce LR when val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        # CSV logger
        keras.callbacks.CSVLogger(
            os.path.join(config.REPORTS_DIR, f"training_phase{phase}_{ts}.csv"),
            separator=",",
            append=False,
        ),
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.RESULTS_DIR, f"tb_phase{phase}_{ts}"),
            histogram_freq=1,
        ),
    ]
    return callbacks


# ══════════════════════════════════════════════════════════════
#  Training entry point
# ══════════════════════════════════════════════════════════════

def train(
    train_dir: str = config.TRAIN_DIR,
    test_dir:  str = config.TEST_DIR,
    epochs_p1: int = config.EPOCHS_FROZEN,
    epochs_p2: int = config.EPOCHS_FINETUNE,
    batch_size: int = config.BATCH_SIZE,
    use_class_weights: bool = True,
):
    """
    Full two-phase training routine.

    Returns
    -------
    model    : trained keras.Model
    history1 : History from phase 1
    history2 : History from phase 2
    """

    logger.info("=" * 60)
    logger.info("  VGG16-NADE  –  Brain Tumour Classification")
    logger.info("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────
    train_gen, val_gen, test_gen = build_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size,
    )

    class_weights = compute_class_weights(train_gen) if use_class_weights else None

    # ── 2. Build model ───────────────────────────────────────
    model = build_vgg16_nade_model()
    print_model_summary(model)

    # ── PHASE 1: Frozen backbone ──────────────────────────────
    logger.info("\n🔒 PHASE 1 — Training head & NADE (backbone frozen)")
    compile_phase1(model)

    history1 = model.fit(
        train_gen,
        epochs=epochs_p1,
        validation_data=val_gen,
        callbacks=make_callbacks(phase=1),
        class_weight=class_weights,
        verbose=1,
    )

    # ── PHASE 2: Fine-tune VGG16 block5 ──────────────────────
    logger.info("\n🔓 PHASE 2 — Fine-tuning VGG16 block5 + head")
    unfreeze_for_finetuning(model)

    history2 = model.fit(
        train_gen,
        epochs=epochs_p2,
        validation_data=val_gen,
        callbacks=make_callbacks(phase=2),
        class_weight=class_weights,
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────
    model.save(config.FINAL_MODEL_PATH)
    logger.info(f"Final model saved → {config.FINAL_MODEL_PATH}")

    # ── Persist combined history ──────────────────────────────
    combined_history = _merge_histories(history1.history, history2.history)
    history_path = os.path.join(config.REPORTS_DIR, "combined_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in combined_history.items()}, f, indent=2)
    logger.info(f"History saved → {history_path}")

    return model, history1, history2


def _merge_histories(h1: dict, h2: dict) -> dict:
    """Concatenate two history dicts (same keys)."""
    merged = {}
    for key in h1:
        merged[key] = h1[key] + h2.get(key, [])
    return merged


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG16-NADE brain-tumour classifier")
    parser.add_argument("--train_dir",  default=config.TRAIN_DIR)
    parser.add_argument("--test_dir",   default=config.TEST_DIR)
    parser.add_argument("--epochs_p1",  type=int, default=config.EPOCHS_FROZEN)
    parser.add_argument("--epochs_p2",  type=int, default=config.EPOCHS_FINETUNE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights,
    )
