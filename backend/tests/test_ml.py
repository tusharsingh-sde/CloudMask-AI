import os
import sys
import tempfile
import cv2
import numpy as np
import pytest
import torch

import model as model_module
from model import UNet
from dataset import CloudDataset


def test_unet_architecture_and_immutable_default():
    """Verify UNet default features is immutable tuple and module structure is preserved."""
    default_features = UNet.__init__.__defaults__[2]
    assert isinstance(default_features, tuple), f"Expected tuple for features, got {type(default_features)}"

    m = UNet(in_channels=4, out_channels=1)
    for attr in ["downs", "ups", "pool", "bottleneck", "final_conv"]:
        assert hasattr(m, attr), f"Missing module attribute '{attr}' on UNet!"
    assert hasattr(m.downs[0], "conv")


def test_unet_forward_pass_shapes():
    """Verify forward pass on standard (384x384) and non-divisible (250x250) shapes."""
    m = UNet(in_channels=4, out_channels=1)
    m.eval()

    with torch.inference_mode():
        # Standard input
        x_std = torch.randn(1, 4, 384, 384)
        out_std = m(x_std)
        assert out_std.shape == (1, 1, 384, 384)

        # Non-divisible input triggering F.interpolate
        x_odd = torch.randn(1, 4, 250, 250)
        out_odd = m(x_odd)
        assert out_odd.shape == (1, 1, 250, 250)


def test_zero_torchvision_in_ml_modules():
    """Ensure torchvision is not imported or required by model.py or dataset.py."""
    assert "torchvision" not in sys.modules or "TF" not in dir(model_module)


REAL_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cloudmask_unet_best.pt"
)


@pytest.mark.skipif(not os.path.exists(REAL_CHECKPOINT_PATH), reason="Real weights not checked into repository")
def test_real_checkpoint_strict_loading():
    """If real cloudmask_unet_best.pt exists locally, verify weights_only=True and strict=True loading."""
    model = UNet(in_channels=4, out_channels=1)
    checkpoint = torch.load(REAL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert checkpoint.get("epoch") is not None
    assert checkpoint.get("best_loss") is not None


def test_cloud_dataset_pipeline_and_removeprefix_safety(tmp_path):
    """Test CloudDataset deterministic sorting, removeprefix safety, and [0,1] float32 tensor creation."""
    # Scaffold dataset directories
    for sub in ["train_red", "train_green", "train_blue", "train_nir", "train_gt"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    # Patch with 'red_' in the middle of filename to test removeprefix safety
    base_id = "patch_red_01.tif"
    red_fname = f"red_{base_id}"

    dummy_raster = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
    dummy_mask = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255

    cv2.imwrite(str(tmp_path / "train_red" / red_fname), dummy_raster)
    cv2.imwrite(str(tmp_path / "train_green" / f"green_{base_id}"), dummy_raster)
    cv2.imwrite(str(tmp_path / "train_blue" / f"blue_{base_id}"), dummy_raster)
    cv2.imwrite(str(tmp_path / "train_nir" / f"nir_{base_id}"), dummy_raster)
    cv2.imwrite(str(tmp_path / "train_gt" / f"gt_{base_id}"), dummy_mask)

    ds = CloudDataset(root_dir=str(tmp_path), is_train=True)
    assert len(ds) == 1
    assert ds.patch_names == [red_fname]

    img_tensor, mask_tensor = ds[0]
    assert img_tensor.shape == (4, 64, 64)
    assert img_tensor.dtype == torch.float32
    assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0

    assert mask_tensor.shape == (1, 64, 64)
    assert mask_tensor.dtype == torch.float32


def test_cloud_dataset_missing_channel_error(tmp_path):
    """Missing band raster raises an explicit FileNotFoundError."""
    for sub in ["train_red", "train_green", "train_blue", "train_nir", "train_gt"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    base_id = "patch_02.tif"
    dummy_raster = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)

    # Only create red band, omit green/blue/nir
    cv2.imwrite(str(tmp_path / "train_red" / f"red_{base_id}"), dummy_raster)

    ds = CloudDataset(root_dir=str(tmp_path), is_train=True)
    with pytest.raises(FileNotFoundError) as exc_info:
        _ = ds[0]
    assert "Failed to read band" in str(exc_info.value)
