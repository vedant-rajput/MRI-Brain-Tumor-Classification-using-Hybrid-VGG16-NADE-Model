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

 
