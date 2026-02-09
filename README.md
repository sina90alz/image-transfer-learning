# 🩺 Image Classification with Transfer Learning

This project implements **image classification using transfer learning** with
pretrained convolutional neural networks (**EfficientNet / ResNet**) in **PyTorch**.

The model classifies chest X-ray images into:

-   NORMAL
-   PNEUMONIA
-   TUBERCULOSIS

The project focuses not only on model performance, but also on **proper
evaluation, metric interpretation, and failure-mode analysis**.

------------------------------------------------------------------------

## Project Overview

- Task: **Multi-class image classification**
- Domain: **Chest X-ray medical imaging**
- Classes:
  - NORMAL
  - PNEUMONIA
  - TUBERCULOSIS
- Framework: **PyTorch**
- Training setup: **CPU-friendly, reproducible**

------------------------------------------------------------------------

## 🚀 Features

-   Programmatic dataset ingestion (Kaggle)
-   Transfer learning with EfficientNet-B0 (ImageNet pretrained)
-   Phase A: Frozen backbone (feature extraction)
-   Phase B: Partial fine-tuning (`features[-3:]`)
-   Weighted Cross-Entropy Loss for class imbalance
-   Data augmentation pipeline
-   Cosine learning rate scheduler
-   Per-class evaluation metrics
-   Confusion matrix analysis

------------------------------------------------------------------------

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

------------------------------------------------------------------------

### Phase A — Feature Extraction (Frozen Backbone)

**Objective:**  
Train a stable classifier head on top of a fixed, pretrained feature extractor.

**What happens in this phase:**
- A CNN backbone pretrained on **ImageNet** is loaded
- All backbone layers are **frozen**
- The final classification layer is replaced to match the dataset classes
- Only the classifier head is trained

### Phase B — Fine-Tuning (Partial Unfreeze)

**Objective:**
- Adapt high-level features of the pretrained model to the medical imaging domain.

**What happens in this phase:**
- The best checkpoint from Phase A is loaded
- Most of the backbone remains frozen
- Only the last block of the network is unfrozen
- Training continues with a lower learning rate

------------------------------------------------------------------------

## 🛠 Setup

``` bash
pip install -r requirements.txt
python scripts/download_raw.py
python scripts/split_dataset.py
python -m src.train
```

------------------------------------------------------------------------

# 📊 Evaluation Metrics & Results

## 🧪 Test Performance

  Metric                          Value
  ------------------------------- ------------
  **Accuracy**                    **79.14%**
  **Macro F1-score**              **0.8020**
  **Weighted F1-score**           **0.7911**
  **Test Loss (Cross-Entropy)**   **0.4084**

------------------------------------------------------------------------

## 📈 Per-Class Performance (Test Set)

  Class              Precision   Recall   F1-score   Support
  ------------------ ----------- -------- ---------- ---------
  **NORMAL**         0.6801      0.7868   0.7296     727
  **PNEUMONIA**      0.8096      0.9722   0.8835     468
  **TUBERCULOSIS**   0.9208      0.6960   0.7928     852

------------------------------------------------------------------------

## 🔎 Confusion Matrix (Test Set)

Rows = True Labels\
Columns = Predicted Labels

                     Predicted
                  NORMAL  PNEUMONIA  TUBERCULOSIS
    TRUE NORMAL      572      107          48
    TRUE PNEUMONIA    10      455           3
    TRUE TB          259        0         593

------------------------------------------------------------------------

# 🧠 Key Observations

-   The model demonstrates strong balanced performance (**Macro F1 ≈
    0.80**).
-   **PNEUMONIA detection is highly reliable** (Recall ≈ 97%).
-   **TUBERCULOSIS precision is high (92%)**, meaning TB predictions are
    trustworthy.
-   The primary limitation is **TB misclassified as NORMAL**, indicating
    a feature separability challenge between these two classes.
-   Weighted cross-entropy loss helped maintain balanced performance
    across classes.

------------------------------------------------------------------------

# 🏁 Conclusion

This project demonstrates:

-   Effective application of transfer learning in medical imaging
-   Controlled fine-tuning of pretrained CNN backbones
-   Proper handling of class imbalance
-   Metric-driven model evaluation and interpretation
-   Identification of real-world failure modes via confusion matrix
    analysis

The model achieves strong and balanced multi-class performance while
maintaining interpretability and clear improvement strategies.

------------------------------------------------------------------------

## 📌 Next Steps

Potential improvements include:

-   Increasing input resolution for better TB feature extraction
-   Further backbone fine-tuning with discriminative learning rates
-   Hard example mining for TB vs NORMAL separation
-   Feature embedding visualization (t-SNE / UMAP) to analyze
    separability