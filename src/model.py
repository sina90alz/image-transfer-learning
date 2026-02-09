import torch
import torch.nn as nn
from torchvision import models

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_name: {model_name}")

def freeze_backbone(model: nn.Module, model_name: str):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze classification head
    if model_name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True

def unfreeze_last_block(model: nn.Module, model_name: str):
    # keep head trainable (already is), and unfreeze last stage
    if model_name == "resnet50":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_b0":
        # EfficientNet: unfreeze last few feature blocks
        for p in model.features[-3:].parameters():
            p.requires_grad = True
