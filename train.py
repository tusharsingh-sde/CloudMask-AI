import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CloudDataset
from model import UNet

# 1. Hardware Speed Boosters
torch.backends.cudnn.benchmark = True  # Auto-tunes convolution algorithms for fixed 384x384 input size

# 2. Auto-detect Dataset Path (Works on both Local PC and Kaggle)
DATA_DIR = "38-Cloud_training"
if not os.path.exists(DATA_DIR):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_red" in dirs:
            DATA_DIR = root
            break

# 3. Turbo Hyperparameters
LEARNING_RATE = 3e-4  # Slightly higher LR for larger batch size
BATCH_SIZE = 32       # 4x larger than before (FP16 + Dual GPU handles this easily)
EPOCHS = 20
NUM_WORKERS = 4 if os.name != 'nt' else 0  # Linux/Kaggle par 4 workers, Windows par 0

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    loop = tqdm(loader, leave=True)
    total_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        # non_blocking=True speeds up RAM to VRAM transfer
        data = data.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        
        # Forward Pass in FP16 (Mixed Precision)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # Faster zero_grad (sets memory to None instead of filling zeros)
        optimizer.zero_grad(set_to_none=True)
        
        # Backward Pass with Scaled Gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Device: {device} | GPUs: {num_gpus} | Dataset: {DATA_DIR}")
    
    model = UNet(in_channels=4, out_channels=1)
    
    # Enable Dual-GPU if running on Kaggle T4 x2
    if num_gpus > 1:
        print(f"🔥 Dual-GPU Activated: Splitting batches across {num_gpus} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    
    train_dataset = CloudDataset(root_dir=DATA_DIR, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE if device == "cuda" else 4, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )
    
    best_loss = float("inf")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        print(f"✅ Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Unwrap DataParallel before saving so api.py can load it normally on CPU
        raw_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": raw_state_dict,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "avg_loss": avg_loss,
            "best_loss": best_loss
        }
        
        # Save per-epoch checkpoint (so no data is ever lost)
        torch.save(checkpoint, f"cloudmask_unet_epoch_{epoch+1}.pt")
        print(f"💾 Epoch {epoch+1} Checkpoint Saved!")
        
        # Save latest (always overwritten for quick resume)
        torch.save(checkpoint, "cloudmask_unet_latest.pt")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, "cloudmask_unet_best.pt")
            print(f"🌟 New Best Model Saved! (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()