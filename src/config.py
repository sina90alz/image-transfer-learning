from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # -------------------------
    # Paths
    # -------------------------
    data_dir: Path = Path("data")          # expects data/train, data/val, data/test
    raw_dir: Path = Path("data/raw")       # where download_raw.py puts images
    output_dir: Path = Path("runs")        # checkpoints / outputs

    # -------------------------
    # Data / Dataloader
    # -------------------------
    image_size: int = 160                  # CPU-friendly (224 is standard)
    batch_size: int = 16                   # CPU-friendly; try 8 if slow
    num_workers: int = 0                   # recommended on Windows + CPU
    pin_memory: bool = False               # only useful with CUDA

    # -------------------------
    # Training
    # -------------------------
    epochs: int = 10                       # first run = sanity check
    seed: int = 42
    device: str = "auto"                   # "auto" | "cpu" | "cuda"

    # -------------------------
    # Model (transfer learning)
    # -------------------------
    model_name: str = "efficientnet_b0"    # "resnet50" or "efficientnet_b0"
    pretrained: bool = True

    # Phase A: feature extraction
    freeze_backbone: bool = True
    # Phase B: fine-tuning
    fine_tune_last_block: bool = True     # set True after Phase A
    resume_from_checkpoint: bool = True
    checkpoint_path: str | None = "runs/best_efficientnet_b0.pt"

    # -------------------------
    # Optimizer
    # -------------------------
    lr: float = 3e-5
    weight_decay: float = 1e-4

    # Optional: gradient clipping (safer/stabler)
    grad_clip_norm: float | None = 1.0     # set None to disable

    # -------------------------
    # LR Scheduler
    # -------------------------
    scheduler: str = "cosine"                # "step" | "cosine"
    step_size: int = 3                     # for StepLR
    gamma: float = 0.1                     # for StepLR
    cosine_t_max: int | None = None        # if None, will use epochs

    # -------------------------
    # Logging / Saving
    # -------------------------
    save_best_only: bool = True
    run_name: str = "baseline_cpu"         # used in checkpoint file name

    def resolve_device(self) -> str:
        """Helper to choose device at runtime."""
        if self.device in ("cpu", "cuda"):
            return self.device

        # auto
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
