import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CloudDataset
from model import UNet

# Hardware Speed Optimization
torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 42):
    """Ensure reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute binary Intersection-over-Union (IoU) and Dice Coefficient.
    predictions: Raw logits (B, 1, H, W)
    targets: Binary ground truth masks (B, 1, H, W)
    """
    with torch.no_grad():
        probs = torch.sigmoid(predictions)
        preds = (probs > threshold).float()
        intersection = (preds * targets).sum().item()
        total_pred = preds.sum().item()
        total_target = targets.sum().item()
        union = total_pred + total_target - intersection

        iou = (intersection + 1e-7) / (union + 1e-7)
        dice = (2.0 * intersection + 1e-7) / (total_pred + total_target + 1e-7)
        return float(iou), float(dice)


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: str,
    max_norm: float = 1.0
) -> Tuple[float, float, float]:
    """Train for one epoch with mixed precision (AMP), gradient clipping, and metric tracking."""
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for data, targets in loop:
        data = data.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        # Forward Pass in FP16 (Mixed Precision)
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale gradients prior to clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        scaler.step(optimizer)
        scaler.update()

        iou, dice = compute_metrics(predictions, targets)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{iou:.3f}",
            dice=f"{dice:.3f}"
        )

    num_batches = max(1, len(loader))
    return (
        float(total_loss / num_batches),
        float(total_iou / num_batches),
        float(total_dice / num_batches)
    )


def validate_fn(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str
) -> Tuple[float, float, float]:
    """Run validation under inference mode without computing gradients."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.inference_mode():
        for data, targets in loader:
            data = data.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            iou, dice = compute_metrics(predictions, targets)

            total_loss += loss.item()
            total_iou += iou
            total_dice += dice

    num_batches = max(1, len(loader))
    return (
        float(total_loss / num_batches),
        float(total_iou / num_batches),
        float(total_dice / num_batches)
    )


def resolve_default_data_dir() -> str:
    """Auto-detect 38-Cloud dataset path across local dev and Kaggle environments."""
    candidates = [
        "38-Cloud_training",
        os.path.join("..", "38-Cloud_training"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "38-Cloud_training"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    for root, dirs, _ in os.walk("/kaggle/input"):
        if "train_red" in dirs:
            return root

    return "38-Cloud_training"


def parse_args():
    parser = argparse.ArgumentParser(description="Train CloudMask AI U-Net on 38-Cloud Multispectral Dataset")
    parser.add_argument("--data-dir", type=str, default=resolve_default_data_dir(), help="Path to 38-Cloud training folder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32 if torch.cuda.is_available() else 4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation set split fraction (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic train/val split")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoint every N epochs (0 = disabled)")
    parser.add_argument("--num-workers", type=int, default=4 if os.name != "nt" else 0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"🚀 Device: {device} | GPUs: {num_gpus} | Dataset: {args.data_dir} | Seed: {args.seed}")

    model = UNet(in_channels=4, out_channels=1)

    # Multi-GPU support if running on distributed hardware
    if num_gpus > 1:
        print(f"🔥 Multi-GPU Activated: DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    start_epoch = 0
    best_loss = float("inf")

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            raw_model.load_state_dict(checkpoint["state_dict"])
            if "optimizer" in checkpoint and checkpoint["optimizer"]:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint and checkpoint["scaler"] and device == "cuda":
                scaler.load_state_dict(checkpoint["scaler"])
            if "scheduler" in checkpoint and checkpoint["scheduler"]:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_loss = float(checkpoint.get("best_loss", float("inf")))
            print(f"   Resumed at epoch {start_epoch + 1} with previous best loss {best_loss:.4f}")
        else:
            print(f"⚠️ Checkpoint file '{args.resume}' not found! Starting fresh training.")

    full_dataset = CloudDataset(root_dir=args.data_dir, is_train=True)
    total_samples = len(full_dataset)
    if total_samples == 0:
        print(f"⚠️ Warning: No training patches found in '{args.data_dir}'. Please verify data directory.")
        return

    # Deterministic Train / Validation Split
    val_size = int(total_samples * args.val_split)
    train_size = total_samples - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📊 Dataset split: {train_size} training patches, {val_size} validation patches ({args.val_split*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0)
    ) if val_size > 0 else None

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch + 1
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n--- Epoch {current_epoch}/{args.epochs} (LR: {current_lr:.6f}) ---")

        train_loss, train_iou, train_dice = train_one_epoch(
            train_loader, model, optimizer, loss_fn, scaler, device, max_norm=1.0
        )
        scheduler.step()

        if val_loader:
            val_loss, val_iou, val_dice = validate_fn(val_loader, model, loss_fn, device)
            eval_metric = val_loss
            print(f"✅ Epoch {current_epoch} | Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} Dice: {val_dice:.4f}")
        else:
            val_loss, val_iou, val_dice = train_loss, train_iou, train_dice
            eval_metric = train_loss
            print(f"✅ Epoch {current_epoch} | Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} Dice: {train_dice:.4f}")

        # Check if validation metric improved and update best_loss BEFORE creating checkpoint dicts
        is_best = eval_metric < best_loss
        if is_best:
            best_loss = float(eval_metric)

        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        # Full resumable checkpoint for training continuation
        resumable_checkpoint = {
            "epoch": int(current_epoch),
            "state_dict": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_iou": float(val_iou),
            "val_dice": float(val_dice),
            "best_loss": float(best_loss)
        }

        # Save latest checkpoint (always overwritten for quick resume)
        torch.save(resumable_checkpoint, "cloudmask_unet_latest.pt")

        # Save periodic epoch checkpoint if enabled
        if args.save_every > 0 and current_epoch % args.save_every == 0:
            epoch_path = f"cloudmask_unet_epoch_{current_epoch}.pt"
            torch.save(resumable_checkpoint, epoch_path)
            print(f"💾 Saved periodic checkpoint: {epoch_path}")

        # Save lightweight inference-only checkpoint when new best is achieved
        if is_best:
            inference_checkpoint = {
                "epoch": int(current_epoch),
                "state_dict": raw_model.state_dict(),
                "best_loss": float(best_loss),
                "val_loss": float(val_loss),
                "val_iou": float(val_iou),
                "val_dice": float(val_dice)
            }
            torch.save(inference_checkpoint, "cloudmask_unet_best.pt")
            print(f"🌟 New Best Model Saved to cloudmask_unet_best.pt! (Val Loss: {best_loss:.4f}, IoU: {val_iou:.4f})")


if __name__ == "__main__":
    main()
