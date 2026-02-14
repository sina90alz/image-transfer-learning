from __future__ import annotations

import json
from pathlib import Path
import torch
import torchvision.models as models


def build_model(num_classes: int = 3) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Adjust to your real weights path/name
    weights_path = project_root / "runs" / "best_efficientnet_b0.pt"
    onnx_path = artifacts_dir / "model.onnx"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    # 1) Load PyTorch model + weights
    model = build_model(num_classes=3)
    ckpt = torch.load(weights_path, map_location="cpu")

    # Your checkpoint format: {"model": state_dict, "classes": [...]}
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 2) Dummy input that matches your API input size
    dummy = torch.randn(1, 3, 160, 160, dtype=torch.float32)

    # 3) Export
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"✅ Exported ONNX -> {onnx_path}")

    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    if classes:
        (artifacts_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
        print(f"✅ Saved classes -> {artifacts_dir / 'classes.json'}")

if __name__ == "__main__":
    main()
