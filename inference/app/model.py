from __future__ import annotations

import io
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ImageNet normalization (must match pretrained backbone expectations)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_model_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)  # weights not needed at inference
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_preprocess(image_size: int):
    # Deterministic transforms for inference
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class InferenceModel:
    def __init__(self, model_path: str, image_size: int, device: str = "cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(model_path, map_location=self.device)
        self.classes: List[str] = ckpt.get("classes")
        if not self.classes:
            raise RuntimeError("Checkpoint missing 'classes'. Re-save checkpoint with classes list.")

        self.model = build_model_efficientnet_b0(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = build_preprocess(image_size=image_size)

    def predict_topk(self, image_bytes: bytes, k: int = 3) -> List[Tuple[str, float]]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        topk = torch.topk(probs, k=min(k, probs.numel()))
        results = []
        for idx, conf in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append((self.classes[idx], float(conf)))
        return results
