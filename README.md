# Image Classification with Transfer Learning

This project demonstrates image classification using **transfer learning**
with pretrained CNNs (ResNet / EfficientNet) in PyTorch.

## Features
- Custom dataset ingestion (Kaggle)
- Reproducible data pipeline (raw → train/val/test)
- Data augmentation
- Frozen backbone + fine-tuning
- Learning rate scheduling

## Setup
```bash
pip install -r requirements.txt
python scripts/download_raw.py
python scripts/split_dataset.py
python -m src.train
