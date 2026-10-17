import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class CloudDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        """
        root_dir: '38-Cloud_training' or '38-Cloud_test' path
        is_train: True for training set, False for testing set
        """
        self.root_dir = root_dir
        self.is_train = is_train
        
        # set directories based on whether it's training or testing
        prefix = 'train' if self.is_train else 'test'
        self.red_dir = os.path.join(root_dir, f'{prefix}_red')
        self.green_dir = os.path.join(root_dir, f'{prefix}_green')
        self.blue_dir = os.path.join(root_dir, f'{prefix}_blue')
        self.nir_dir = os.path.join(root_dir, f'{prefix}_nir')
        self.gt_dir = os.path.join(root_dir, f'{prefix}_gt') if self.is_train else None
        
        # extracting patch names from the red channel directory
        # README: Assuming that all channels have the same number of patches and the same naming convention, we can just list the red channel directory to get the patch names.
        self.patch_names = [f for f in os.listdir(self.red_dir) if f.endswith(('.TIF', '.tif', '.jpg'))]

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        file_name = self.patch_names[idx]
        
        # Extract the base ID to find matching green, blue, nir, gt files
        base_id = file_name.replace('red_', '')
        
        # Paths for all 4 channels
        red_path = os.path.join(self.red_dir, file_name)
        green_path = os.path.join(self.green_dir, 'green_' + base_id)
        blue_path = os.path.join(self.blue_dir, 'blue_' + base_id)
        nir_path = os.path.join(self.nir_dir, 'nir_' + base_id)
        
        r = cv2.imread(red_path, 0)
        g = cv2.imread(green_path, 0)
        b = cv2.imread(blue_path, 0)
        nir = cv2.imread(nir_path, 0)
        
        # Stack all 4 channels to create a single multi-channel image
        img_stacked = np.stack([r, g, b, nir], axis=-1)
        
        # Convert to PyTorch Tensor -> Shape (4, H, W)
        image_tensor = TF.to_tensor(img_stacked)
        
        # if we are in training mode, load the ground truth mask
        if self.is_train:
            gt_path = os.path.join(self.gt_dir, 'gt_' + base_id)
            mask = cv2.imread(gt_path, 0)
            # mask to binary float32 tensor
            mask = (mask > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0) # Shape: (1, H, W)
            return image_tensor, mask_tensor
        else:
            return image_tensor, file_name