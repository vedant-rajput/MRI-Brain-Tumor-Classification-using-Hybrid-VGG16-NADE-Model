"""
utils.py
────────
Shared utility functions:
  • Reproducibility seeding
  • GPU configuration
  • Dataset download helper (Kaggle API)
  • Directory setup
  • Logging setup
"""

import os
import sys
import logging
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int = config.RANDOM_SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global seed set to {seed}")


# ──────────────────────────────────────────────
# GPU configuration
# ──────────────────────────────────────────────

def configure_gpu(memory_limit_mb: int = None, allow_growth: bool = True):
    """
    Configure TensorFlow GPU settings.

    Parameters
    ──────────
    memory_limit_mb : cap GPU memory usage (None = unlimited)
    allow_growth    : dynamically allocate GPU memory
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected — running on CPU")
        return

    try:
        for gpu in gpus:
            if allow_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )],
                )
        logger.info(f"GPU(s) configured: {[g.name for g in gpus]}")
    except RuntimeError as e:
        logger.warning(f"GPU config failed (must be set before ops): {e}")


# ──────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────

def ensure_dirs():
    """Create all required project directories."""
    dirs = [
        config.RAW_DATA_DIR,
        config.PROC_DATA_DIR,
        config.MODEL_DIR,
        config.PLOTS_DIR,
        config.REPORTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Project directories ensured.")


# ──────────────────────────────────────────────
# Kaggle dataset download
# ──────────────────────────────────────────────

def download_dataset(
    kaggle_dataset: str = "masoudnickparvar/brain-tumor-mri-dataset",
    dest_dir: str = config.RAW_DATA_DIR,
):
    """
    Download dataset from Kaggle using the kaggle CLI.
    Requires ~/.kaggle/kaggle.json with valid credentials.

    Dataset structure after unzip:
        raw/
          Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
          Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError("Install kaggle: `pip install kaggle`")

    os.makedirs(dest_dir, exist_ok=True)
    cmd = (
        f"kaggle datasets download -d {kaggle_dataset} "
        f"--path {dest_dir} --unzip"
    )
    logger.info(f"Downloading dataset: {kaggle_dataset}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(
            "Kaggle download failed. Ensure ~/.kaggle/kaggle.json exists "
            "and the kaggle package is installed."
        )
    logger.info(f"Dataset downloaded to {dest_dir}")


# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

def setup_logging(level: int = logging.INFO, log_file: str = None):
    """Configure root logger with optional file output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=handlers,
    )


# ──────────────────────────────────────────────
# TensorFlow/Keras version info
# ──────────────────────────────────────────────

def print_env_info():
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  TensorFlow  : {tf.__version__}")
    print(f"  Keras       : {tf.keras.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPUs        : {len(gpus)}  ({[g.name for g in gpus]})")
