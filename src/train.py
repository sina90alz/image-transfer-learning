import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# NEW: metrics
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.data import build_loaders
from src.model import build_model, freeze_backbone, unfreeze_last_block


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    """Returns (avg_loss, avg_acc) using the provided loss_fn."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += bs

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_with_report(model, loader, device, loss_fn, class_names, title="Eval"):
    """
    Prints:
      - loss/acc
      - per-class precision/recall/F1
      - confusion matrix
    Returns (avg_loss, avg_acc).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        preds = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (preds == y).sum().item()
        n += bs

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / n
    avg_acc = correct / n

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    print(f"\n{title}: loss={avg_loss:.4f} acc={avg_acc:.4f}")
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    return avg_loss, avg_acc


def make_scheduler(cfg, optimizer):
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def main():
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, test_loader, classes = build_loaders(
        cfg.data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers
    )
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(cfg.model_name, num_classes, pretrained=cfg.pretrained)

    # Phase setup
    if cfg.freeze_backbone:
        freeze_backbone(model, cfg.model_name)
    if cfg.fine_tune_last_block:
        unfreeze_last_block(model, cfg.model_name)

    model = model.to(device)

    # Weighted loss from train dataset (used for BOTH training and evaluation)
    counts = Counter(train_loader.dataset.targets)  # class_id -> count
    weights = torch.tensor([1.0 / counts[i] for i in range(num_classes)], dtype=torch.float)
    weights = weights / weights.mean()  # normalize to keep scale stable
    weights = weights.to(device)

    print("Class counts:", counts)
    print("Class weights:", weights.detach().cpu().tolist())

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Resume model weights (Phase B uses Phase A best checkpoint)
    if cfg.resume_from_checkpoint and cfg.checkpoint_path:
        print(f"Resuming from checkpoint: {cfg.checkpoint_path}")
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    # Only train parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(cfg, optimizer)

    best_val_acc = 0.0
    best_path = cfg.output_dir / f"best_{cfg.model_name}.pt"

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(logits, y) * bs
            n += bs

            pbar.set_postfix(
                loss=running_loss / n,
                acc=running_acc / n,
                lr=optimizer.param_groups[0]["lr"],
            )

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        print(f"  Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(f"Saved best model to {best_path} (val_acc={best_val_acc:.4f})")

            # OPTIONAL: print per-class metrics when a new best is found
            # Comment out if you find it too verbose.
            evaluate_with_report(
                model, val_loader, device, loss_fn, classes, title="Val (BEST)"
            )

    # Final test with best weights
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Print full metrics on test set
    evaluate_with_report(model, test_loader, device, loss_fn, classes, title="Test")


if __name__ == "__main__":
    main()
