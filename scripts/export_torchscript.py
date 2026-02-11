import torch
from torchvision import models
import torch.nn as nn
from pathlib import Path

MODEL_NAME = "efficientnet_b0"
CKPT_PATH = Path("runs/best_efficientnet_b0.pt")
OUT_PATH = Path("artifacts/efficientnet_b0_torchscript.pt")
IMAGE_SIZE = 160

def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Trace with example input (batch=1)
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, example)
    traced.save(str(OUT_PATH))

    # Save classes alongside for deployment
    (OUT_PATH.parent / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    print(f"✅ Saved TorchScript model to: {OUT_PATH}")
    print(f"✅ Saved classes to: {OUT_PATH.parent / 'classes.txt'}")

if __name__ == "__main__":
    main()
