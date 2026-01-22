from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    data_dir: Path = Path("data")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 10

    model_name: str = "resnet50"   # "resnet50" or "efficientnet_b0"
    pretrained: bool = True

    freeze_backbone: bool = True   # first phase
    fine_tune_last_block: bool = False  # second phase

    lr: float = 3e-4
    weight_decay: float = 1e-4

    scheduler: str = "cosine"  # "cosine" or "step"
    step_size: int = 5
    gamma: float = 0.1

    seed: int = 42
    output_dir: Path = Path("runs")
