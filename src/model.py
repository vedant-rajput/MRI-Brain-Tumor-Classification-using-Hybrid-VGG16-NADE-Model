"""
model.py
────────
Hybrid VGG16-NADE Architecture for MRI Brain Tumor Classification.

Architecture Overview
─────────────────────
  Input (224×224×3)
       │
  ┌────▼────────────────────────────────┐
  │  VGG16 Backbone (ImageNet weights)  │
  │  Frozen during phase-1 training     │
  │  Fine-tuned from block5 in phase-2  │
  └────┬────────────────────────────────┘
       │  Feature maps  (7×7×512)
       │
  ┌────▼──────────────────────────────────────┐
  │  Global Average Pooling  →  (512,)        │
  └────┬──────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────────────┐
  │  NADE Module  (Neural Autoregressive Density Est.) │
  │  ─ Projects to NADE_INPUT_DIM features             │
  │  ─ NADE_STEPS autoregressive passes               │
  │    each conditioned on mask of prior dims         │
  │  ─ Outputs enhanced representation + log-density  │
  └────┬──────────────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────┐
  │  Classification Head                       │
  │  Dense(256, relu) → BN → Dropout(0.4)     │
  │  Dense(128, relu) → BN → Dropout(0.3)     │
  │  Dense(4, softmax)                         │
  └────────────────────────────────────────────┘

NADE Intuition
──────────────
  NADE (Larochelle & Murray, 2011) models the joint distribution
  p(x) = ∏_d p(x_d | x_{1:d-1}) autoregressively.
  In our hybrid model the NADE module applies NADE_STEPS masked
  linear projections over the VGG16 feature vector, progressively
  conditioning each feature subset on its predecessors.
  This produces:
    • A density-regularised representation that improves
      generalisation on small / imbalanced medical datasets.
    • An auxiliary log-density loss term that acts as a
      second regulariser alongside cross-entropy.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  NADE Module  (custom Keras layer)
# ══════════════════════════════════════════════════════════════

class NADELayer(layers.Layer):
    """
    Neural Autoregressive Distribution Estimator layer.

    Given an input vector x ∈ ℝ^D, performs K autoregressive
    conditioning steps:

        for k in 1..K:
            mask_k  ← ones for dims 0..(k*D//K - 1), else zeros
            h_k     ← sigmoid( W_k · (x ⊙ mask_k) + b_k )
            a_k     ← sigmoid( V_k · h_k + c_k )   [reconstruction]

    The mean of all h_k tensors is returned as the NADE representation.
    The auxiliary NADE loss is the mean binary cross-entropy between
    each a_k and the corresponding masked input segment.

    Parameters
    ──────────
    input_dim   : dimensionality of the feature input
    hidden_units: hidden units per autoregressive step
    steps       : number of autoregressive conditioning steps (K)
    """

    def __init__(self, input_dim: int, hidden_units: int, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim    = input_dim
        self.hidden_units = hidden_units
        self.steps        = steps

        # One set of W/b/V/c matrices per autoregressive step
        self.W = []
        self.b = []
        self.V = []
        self.c = []

    def build(self, input_shape):
        D = self.input_dim
        H = self.hidden_units

        for k in range(self.steps):
            # Encoder weights: ℝ^D → ℝ^H
            self.W.append(
                self.add_weight(name=f"W_{k}", shape=(D, H),
                                initializer="glorot_uniform", trainable=True)
            )
            self.b.append(
                self.add_weight(name=f"b_{k}", shape=(H,),
                                initializer="zeros", trainable=True)
            )
            # Decoder weights: ℝ^H → ℝ^D
            self.V.append(
                self.add_weight(name=f"V_{k}", shape=(H, D),
                                initializer="glorot_uniform", trainable=True)
            )
            self.c.append(
                self.add_weight(name=f"c_{k}", shape=(D,),
                                initializer="zeros", trainable=True)
            )
        super().build(input_shape)

    def call(self, x, training=False):
        D            = self.input_dim
        step_size    = D // self.steps
        h_list       = []
        recon_list   = []
        target_list  = []

        # Normalise input to [0,1] for BCE density loss
        x_norm = tf.sigmoid(x)

        for k in range(self.steps):
            # Build binary mask: visible dims = 0..(k+1)*step_size - 1
            cutoff = min((k + 1) * step_size, D)
            mask   = tf.concat([
                tf.ones([cutoff]),
                tf.zeros([D - cutoff])
            ], axis=0)  # (D,)

            masked_x = x_norm * mask                   # (batch, D)
            h_k      = tf.sigmoid(masked_x @ self.W[k] + self.b[k])  # (batch, H)
            a_k      = tf.sigmoid(h_k @ self.V[k] + self.c[k])       # (batch, D)

            h_list.append(h_k)
            recon_list.append(a_k)
            # Target for BCE: the next segment (not yet visible)
            target_list.append(x_norm)

        # NADE representation: mean of all hidden activations
        nade_repr = tf.reduce_mean(tf.stack(h_list, axis=1), axis=1)  # (batch, H)

        # Auxiliary density loss (mean BCE across all steps)
        bce   = keras.losses.BinaryCrossentropy(reduction="none")
        total_loss = tf.reduce_mean(
            tf.stack([
                tf.reduce_mean(bce(target_list[k], recon_list[k]))
                for k in range(self.steps)
            ])
        )
        self.add_loss(0.01 * total_loss)   # weighted density regulariser

        return nade_repr

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(input_dim=self.input_dim,
                        hidden_units=self.hidden_units,
                        steps=self.steps))
        return cfg


# ══════════════════════════════════════════════════════════════
#  Feature Projection Layer
# ══════════════════════════════════════════════════════════════

class FeatureProjection(layers.Layer):
    """
    Projects the raw VGG16 GAP output to NADE_INPUT_DIM with
    L2 normalisation, making the NADE density estimates
    scale-invariant.
    """

    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense      = layers.Dense(output_dim, use_bias=True)
        self.bn         = layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.dense(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = tf.math.l2_normalize(x, axis=-1)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(output_dim=self.output_dim))
        return cfg


# ══════════════════════════════════════════════════════════════
#  Build the Hybrid Model
# ══════════════════════════════════════════════════════════════

def build_vgg16_nade_model(
    input_shape:   tuple = config.INPUT_SHAPE,
    num_classes:   int   = config.NUM_CLASSES,
    nade_input_dim: int  = config.NADE_INPUT_DIM,
    nade_hidden:   int   = config.NADE_HIDDEN_UNITS,
    nade_steps:    int   = config.NADE_STEPS,
    dropout_1:     float = 0.4,
    dropout_2:     float = 0.3,
) -> keras.Model:
    """
    Construct and return the VGG16-NADE hybrid model (not yet compiled).
    """

    # ── VGG16 Backbone ──────────────────────────────────────────
    vgg16_base = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    vgg16_base.trainable = False   # frozen until fine-tune phase
    logger.info(f"VGG16 loaded — {len(vgg16_base.layers)} layers (all frozen)")

    # ── Build Graph ─────────────────────────────────────────────
    inputs        = keras.Input(shape=input_shape, name="mri_input")

    # Pass through frozen VGG16
    x             = vgg16_base(inputs, training=False)            # (B,7,7,512)

    # Global Average Pooling → dense vector
    x             = layers.GlobalAveragePooling2D(name="gap")(x)  # (B,512)

    # Project to NADE input dimension + L2-normalise
    x_proj        = FeatureProjection(nade_input_dim, name="feat_proj")(x)  # (B, NADE_INPUT_DIM)

    # NADE autoregressive module
    nade_repr     = NADELayer(
        input_dim=nade_input_dim,
        hidden_units=nade_hidden,
        steps=nade_steps,
        name="nade",
    )(x_proj)                                                      # (B, nade_hidden)

    # Residual connection: concatenate original projection + NADE repr
    combined      = layers.Concatenate(name="nade_residual")([x_proj, nade_repr])  # (B, NADE_INPUT_DIM + nade_hidden)

    # Classification head
    z             = layers.Dense(256, activation="relu", name="fc1")(combined)
    z             = layers.BatchNormalization(name="bn1")(z)
    z             = layers.Dropout(dropout_1, name="drop1")(z)

    z             = layers.Dense(128, activation="relu", name="fc2")(z)
    z             = layers.BatchNormalization(name="bn2")(z)
    z             = layers.Dropout(dropout_2, name="drop2")(z)

    outputs       = layers.Dense(num_classes, activation="softmax", name="predictions")(z)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VGG16_NADE")
    logger.info(f"Model built — total params: {model.count_params():,}")
    return model


# ══════════════════════════════════════════════════════════════
#  Compile helpers
# ══════════════════════════════════════════════════════════════

def compile_phase1(model: keras.Model, lr: float = config.LEARNING_RATE):
    """Compile for phase-1 training (VGG16 frozen)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    logger.info(f"Phase-1 compiled — LR={lr}")


def unfreeze_for_finetuning(
    model: keras.Model,
    unfreeze_from: str = config.VGG16_UNFREEZE_FROM,
    lr: float = config.FINETUNE_LR,
):
    """
    Unfreeze VGG16 layers from `unfreeze_from` onward and
    recompile with a smaller learning rate for fine-tuning.
    """
    vgg16_base = model.get_layer("vgg16")
    trainable  = False
    for layer in vgg16_base.layers:
        if layer.name == unfreeze_from:
            trainable = True
        layer.trainable = trainable

    n_trainable = sum(1 for l in vgg16_base.layers if l.trainable)
    logger.info(f"Fine-tune: {n_trainable} VGG16 layers now trainable (from {unfreeze_from})")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    logger.info(f"Fine-tune compiled — LR={lr}")


# ══════════════════════════════════════════════════════════════
#  Model summary utility
# ══════════════════════════════════════════════════════════════

def print_model_summary(model: keras.Model):
    """Print layerwise summary with trainable param counts."""
    model.summary(line_length=100)
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    non_trainable_params = sum(
        tf.size(w).numpy() for w in model.non_trainable_weights
    )
    print(f"\n  Trainable params     : {trainable_params:>12,}")
    print(f"  Non-trainable params : {non_trainable_params:>12,}")
    print(f"  Total params         : {trainable_params + non_trainable_params:>12,}\n")
