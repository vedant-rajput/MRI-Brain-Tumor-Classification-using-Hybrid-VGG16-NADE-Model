"""
data_preprocessing.py
─────────────────────
Data loading, augmentation, and tf.data pipeline
for the MRI Brain Tumor dataset.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1.  Keras ImageDataGenerators  (fast path)
# ─────────────────────────────────────────────────────────────

def build_generators(
    train_dir: str = config.TRAIN_DIR,
    test_dir:  str = config.TEST_DIR,
    img_size:  tuple = (config.IMG_HEIGHT, config.IMG_WIDTH),
    batch_size: int  = config.BATCH_SIZE,
    val_split:  float = config.VALIDATION_SPLIT,
    seed: int = config.RANDOM_SEED,
):
    """
    Returns (train_gen, val_gen, test_gen) Keras generators with
    VGG16-style preprocessing and strong augmentation on the train split.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        validation_split=val_split,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        validation_split=val_split,
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    )

    common_kwargs = dict(
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        seed=seed,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir, subset="training", shuffle=True, **common_kwargs
    )
    val_gen = val_datagen.flow_from_directory(
        train_dir, subset="validation", shuffle=False, **common_kwargs
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, shuffle=False, **common_kwargs
    )

    logger.info(
        f"Generators ready — train: {train_gen.samples}, "
        f"val: {val_gen.samples}, test: {test_gen.samples}"
    )
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────
# 2.  tf.data pipeline  (GPU-optimised, optional)
# ─────────────────────────────────────────────────────────────

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor):
    img  = tf.io.read_file(path)
    img  = tf.image.decode_jpeg(img, channels=3)
    img  = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])
    img  = tf.keras.applications.vgg16.preprocess_input(img)
    return img, label


def _augment(img: tf.Tensor, label: tf.Tensor):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # Random rotation via tfa or manual crop-pad
    img = tf.image.random_crop(
        tf.image.resize_with_crop_or_pad(img, config.IMG_HEIGHT + 20, config.IMG_WIDTH + 20),
        [config.IMG_HEIGHT, config.IMG_WIDTH, 3],
    )
    return img, label


def build_tf_dataset(
    directory: str,
    batch_size: int = config.BATCH_SIZE,
    augment:    bool = False,
    val_split:  float = 0.0,
    seed: int = config.RANDOM_SEED,
):
    """
    Builds a tf.data.Dataset from an image directory.
    Returns (dataset,) or (train_ds, val_ds) if val_split > 0.
    """
    paths, labels = [], []
    for idx, cls in enumerate(config.CLASS_NAMES):
        cls_dir = Path(directory) / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for fp in cls_dir.glob(ext):
                paths.append(str(fp))
                labels.append(idx)

    paths  = np.array(paths)
    labels = tf.keras.utils.to_categorical(labels, num_classes=config.NUM_CLASSES)

    if val_split > 0:
        X_tr, X_v, y_tr, y_v = train_test_split(
            paths, labels, test_size=val_split, stratify=labels.argmax(1), random_state=seed
        )
        train_ds = _make_ds(X_tr, y_tr, batch_size, augment=True)
        val_ds   = _make_ds(X_v,  y_v,  batch_size, augment=False)
        return train_ds, val_ds

    return _make_ds(paths, labels, batch_size, augment=augment)


def _make_ds(paths, labels, batch_size, augment):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────
# 3.  Class-weight computation (handles imbalance)
# ─────────────────────────────────────────────────────────────

def compute_class_weights(train_generator) -> dict:
    """Compute balanced class weights from a Keras generator."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_generator.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    weight_dict = {i: w for i, w in enumerate(weights)}
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict


# ─────────────────────────────────────────────────────────────
# 4.  Single-image preprocessing helper  (inference)
# ─────────────────────────────────────────────────────────────

def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single MRI image for model inference.
    Returns array of shape (1, H, W, 3).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img = img.astype(np.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return np.expand_dims(img, axis=0)


# ─────────────────────────────────────────────────────────────
# 5.  Dataset statistics
# ─────────────────────────────────────────────────────────────

def print_dataset_stats(train_dir: str = config.TRAIN_DIR,
                        test_dir:  str = config.TEST_DIR):
    """Print per-class image counts for training and test sets."""
    for split, directory in [("TRAIN", train_dir), ("TEST", test_dir)]:
        print(f"\n{'─'*40}")
        print(f"  {split} — {directory}")
        print(f"{'─'*40}")
        total = 0
        for cls in config.CLASS_NAMES:
            cls_path = Path(directory) / cls
            if cls_path.exists():
                count = len(list(cls_path.glob("*")))
                total += count
                print(f"  {cls:<15}: {count:>5} images")
        print(f"  {'TOTAL':<15}: {total:>5} images")
