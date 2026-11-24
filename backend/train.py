import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CloudDataset
from model import UNet

# Hardware Speed Optimization
torch.backends.cudnn.benchmark = True


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
    device: str
) -> Tuple[float, float, float]:
    """Train for one epoch with mixed precision (AMP) and metric tracking."""
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

    # Kaggle input paths
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
    parser.add_argument("--num-workers", type=int, default=4 if os.name != "nt" else 0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"🚀 Device: {device} | GPUs: {num_gpus} | Dataset: {args.data_dir} | Batch Size: {args.batch_size}")

    model = UNet(in_channels=4, out_channels=1)

    # Multi-GPU support if running on distributed hardware
    if num_gpus > 1:
        print(f"🔥 Multi-GPU Activated: DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    train_dataset = CloudDataset(root_dir=args.data_dir, is_train=True)
    if len(train_dataset) == 0:
        print(f"⚠️ Warning: No training patches found in '{args.data_dir}'. Please verify data directory.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0)
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg_loss, avg_iou, avg_dice = train_one_epoch(
            train_loader, model, optimizer, loss_fn, scaler, device
        )
        print(f"✅ Epoch {epoch + 1} - Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f}")

        # Unwrap DataParallel before saving for standalone CPU/GPU loading
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        checkpoint = {
            "epoch": int(epoch + 1),
            "state_dict": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "avg_loss": float(avg_loss),
            "avg_iou": float(avg_iou),
            "best_loss": float(best_loss)
        }

        # Save per-epoch checkpoint
        epoch_path = f"cloudmask_unet_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, epoch_path)

        # Save latest checkpoint
        torch.save(checkpoint, "cloudmask_unet_latest.pt")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = float(avg_loss)
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, "cloudmask_unet_best.pt")
            print(f"🌟 New Best Model Saved! (Loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
