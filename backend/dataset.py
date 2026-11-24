import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    """
    PyTorch Dataset for 38-Cloud multispectral satellite imagery.
    Loads 4 spectral bands (Red, Green, Blue, NIR) and optional ground truth mask.
    Transforms uint8 imagery directly into normalized float32 tensors without torchvision.
    """

    def __init__(self, root_dir: str, is_train: bool = True):
        """
        root_dir: Path to '38-Cloud_training' or '38-Cloud_test' directory.
        is_train: True for training/validation set (with GT masks), False for inference test set.
        """
        self.root_dir = root_dir
        self.is_train = is_train

        prefix = "train" if self.is_train else "test"
        self.red_dir = os.path.join(root_dir, f"{prefix}_red")
        self.green_dir = os.path.join(root_dir, f"{prefix}_green")
        self.blue_dir = os.path.join(root_dir, f"{prefix}_blue")
        self.nir_dir = os.path.join(root_dir, f"{prefix}_nir")
        self.gt_dir = os.path.join(root_dir, f"{prefix}_gt") if self.is_train else None

        if not os.path.isdir(self.red_dir):
            self.patch_names = []
            return

        # Deterministically collect all valid image patch filenames
        self.patch_names = sorted([
            f for f in os.listdir(self.red_dir)
            if f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
        ])

    def __len__(self) -> int:
        return len(self.patch_names)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, str]]:
        file_name = self.patch_names[idx]
        base_id = file_name.removeprefix("red_")

        # File paths for all 4 multispectral channels
        red_path = os.path.join(self.red_dir, file_name)
        green_path = os.path.join(self.green_dir, "green_" + base_id)
        blue_path = os.path.join(self.blue_dir, "blue_" + base_id)
        nir_path = os.path.join(self.nir_dir, "nir_" + base_id)

        r = cv2.imread(red_path, 0)
        g = cv2.imread(green_path, 0)
        b = cv2.imread(blue_path, 0)
        nir = cv2.imread(nir_path, 0)

        if r is None or g is None or b is None or nir is None:
            missing = [
                name for name, img in [("red", r), ("green", g), ("blue", b), ("nir", nir)]
                if img is None
            ]
            raise FileNotFoundError(f"Failed to read band(s) {missing} for patch '{base_id}' in '{self.root_dir}'.")

        target_shape = r.shape
        if g.shape != target_shape:
            g = cv2.resize(g, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        if b.shape != target_shape:
            b = cv2.resize(b, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        if nir.shape != target_shape:
            nir = cv2.resize(nir, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        # Stack into (H, W, 4) uint8 array
        img_stacked = np.stack([r, g, b, nir], axis=-1)

        # Direct PyTorch tensor conversion: (H, W, 4) uint8 -> (4, H, W) float32 in [0.0, 1.0]
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(img_stacked.transpose(2, 0, 1))
        ).float().div_(255.0)

        # Ground truth mask processing for training
        if self.is_train:
            if not self.gt_dir:
                raise ValueError("Ground truth directory is not configured for training mode.")
            gt_path = os.path.join(self.gt_dir, "gt_" + base_id)
            mask = cv2.imread(gt_path, 0)
            if mask is None:
                raise FileNotFoundError(f"Ground truth mask missing for patch '{base_id}' at '{gt_path}'.")

            if mask.shape != target_shape:
                mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_binary = (mask > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_binary)).unsqueeze(0)  # (1, H, W)
            return image_tensor, mask_tensor
        else:
            return image_tensor, file_name
