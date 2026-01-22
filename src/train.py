import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from data import build_loaders
from model import build_model, freeze_backbone, unfreeze_last_block

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

def make_scheduler(cfg, optimizer):
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
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

    # Only train parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(cfg, optimizer)
    loss_fn = nn.CrossEntropyLoss()

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

            pbar.set_postfix(loss=running_loss/n, acc=running_acc/n, lr=optimizer.param_groups[0]["lr"])

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"  Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(f"Saved best model to {best_path} (val_acc={best_val_acc:.4f})")

    # Final test with best weights
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
