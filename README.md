# Image Classification with Transfer Learning

This project implements **image classification using transfer learning** with
pretrained convolutional neural networks (**EfficientNet / ResNet**) in **PyTorch**.

The goal is to build a **clean, reproducible, and realistic ML pipeline** that follows
best practices used in real-world computer vision projects, with a strong focus on
**staged transfer learning**.

---

## Project Overview

- Task: **Multi-class image classification**
- Domain: **Chest X-ray medical imaging**
- Classes:
  - NORMAL
  - PNEUMONIA
  - TUBERCULOSIS
- Framework: **PyTorch**
- Training setup: **CPU-friendly, reproducible**

---

## Features

- Programmatic dataset ingestion (Kaggle)
- Clean data pipeline:
  - `data/raw` → `data/train / val / test`
- Data augmentation and ImageNet normalization
- Transfer learning with pretrained CNN backbones
- **Two-phase training strategy**
- Learning rate scheduling
- Validation-based checkpointing

---

## Dataset Pipeline

The dataset is downloaded automatically and handled in two distinct steps:

1. **Download & ingestion**
   - Dataset is downloaded via `kagglehub`
   - Images are copied into `data/raw/<class_name>/`

2. **Dataset preparation**
   - Raw images are split into:
     - `data/train`
     - `data/val`
     - `data/test`
   - Class folders are preserved for compatibility with `torchvision.datasets.ImageFolder`

This separation ensures reproducibility and allows re-splitting the dataset without
re-downloading the data.

---

## Transfer Learning Strategy

Instead of training the entire model at once, this project uses a **two-phase transfer
learning strategy**, which is the standard approach in professional computer vision
pipelines.

---

### Phase A — Feature Extraction (Frozen Backbone)

**Objective:**  
Train a stable classifier head on top of a fixed, pretrained feature extractor.

**What happens in this phase:**
- A CNN backbone pretrained on **ImageNet** is loaded
- All backbone layers are **frozen**
- The final classification layer is replaced to match the dataset classes
- Only the classifier head is trained

**Why this is important:**
- ImageNet features already capture generic visual patterns
- Freezing the backbone:
  - prevents catastrophic forgetting
  - stabilizes early training
  - reduces overfitting
- The classifier head learns how to map pretrained features to medical classes

**Typical configuration (Phase A):**
- freeze_backbone = True
- fine_tune_last_block = False
- epochs = 5
- lr = 3e-4
- scheduler = "step"

**Outcome:**
- Stable convergence
- A strong baseline model
- Best checkpoint saved based on validation accuracy
- This checkpoint is reused in Phase B

### Phase B — Fine-Tuning (Partial Unfreeze)

**Objective:**
- Adapt high-level features of the pretrained model to the medical imaging domain.

**What happens in this phase:**
- The best checkpoint from Phase A is loaded
- Most of the backbone remains frozen
- Only the last block of the network is unfrozen
- Training continues with a lower learning rate

**Why this is important:**
- Medical images differ significantly from natural images
- Fine-tuning allows the model to adapt high-level representations

**Unfreezing only the last block:**
- limits overfitting
- preserves generic low-level features
- improves class discrimination safely

**Typical configuration (Phase A):**
- resume_from_checkpoint = True
- checkpoint_path = "runs/best_efficientnet_b0.pt"

- freeze_backbone = True
- fine_tune_last_block = True
- epochs = 8
- lr = 1e-4
- scheduler = "cosine"
