"""Training loop: cosine LR, early stopping, AMP, CUDA."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_loaders
from .model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return total_loss / n, correct / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train_loader, val_loader, _, classes = build_loaders(
        args.data_root, batch_size=args.batch_size,
        augment=not args.no_augment, num_workers=args.num_workers,
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={len(classes)}")

    model = build_model(args.arch, num_classes=len(classes), frozen=args.frozen).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    if args.optimizer == "adamw":
        optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(device="cuda") if device.type == "cuda" and args.amp else None

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    tag = f"{args.arch}{'_frozen' if args.frozen else ''}{'_noaug' if args.no_augment else ''}"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_sec": []}
    best_val_acc, wait = 0.0, 0
    for epoch in range(args.epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        with torch.no_grad():
            vl, va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)
        history["lr"].append(scheduler.get_last_lr()[0]); history["epoch_sec"].append(dt)
        print(f"epoch {epoch+1}/{args.epochs} "
              f"train_loss={tl:.4f} train_acc={ta:.4f} "
              f"val_loss={vl:.4f} val_acc={va:.4f} time={dt:.1f}s")

        if va > best_val_acc:
            best_val_acc, wait = va, 0
            torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": args.arch},
                       out_dir / f"{tag}_best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stop @ epoch {epoch+1}; best val_acc={best_val_acc:.4f}")
                break

    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val_acc={best_val_acc:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "vit_b_16", "efficientnet_b3"])
    p.add_argument("--data_root", default="data/food-101")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
