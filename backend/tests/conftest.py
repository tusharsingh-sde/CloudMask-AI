import io
import os
import sys
import numpy as np
from PIL import Image
import pytest
import tifffile
import torch
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import api
from model import UNet


def make_png_bytes(width: int = 64, height: int = 64, color: int = 120) -> bytes:
    """Generate in-memory 8-bit single-channel PNG bytes."""
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_16bit_tiff_bytes(width: int = 64, height: int = 64, min_val: int = 500, max_val: int = 1500) -> bytes:
    """Generate in-memory 16-bit single-channel GeoTIFF/TIFF bytes."""
    arr = np.random.randint(min_val, max_val, size=(height, width), dtype=np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def make_oversized_png_bytes() -> bytes:
    """Generate in-memory PNG with header exceeding MAX_DIMENSION (5000x200)."""
    img = Image.new("L", (5000, 200), color=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_too_small_png_bytes() -> bytes:
    """Generate in-memory PNG with dimensions below minimum 16x16 (8x8)."""
    img = Image.new("L", (8, 8), color=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_corrupt_bytes() -> bytes:
    """Generate invalid corrupt file bytes."""
    return b"NOT_A_VALID_IMAGE_DATA_123456789"


@pytest.fixture(scope="session")
def dummy_checkpoint_path(tmp_path_factory) -> str:
    """Create a temporary valid lightweight UNet checkpoint for tests that works in CI without downloading weights."""
    tmp_dir = tmp_path_factory.mktemp("models")
    ckpt_path = str(tmp_dir / "cloudmask_unet_test.pt")

    model = UNet(in_channels=4, out_channels=1)
    checkpoint = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "best_loss": 0.05,
        "val_loss": 0.05,
        "val_iou": 0.85,
        "val_dice": 0.90
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


@pytest.fixture
def client(dummy_checkpoint_path, monkeypatch):
    """
    TestClient fixture configured with a temporary valid model and rate limiting disabled by default.
    Ensures FastAPI lifespan executes correctly.
    """
    monkeypatch.setenv("MODEL_PATH", dummy_checkpoint_path)
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCE", "4")

    # Reset in-memory rate limiter state
    api.rate_limiter.reset()

    with TestClient(api.app) as test_client:
        yield test_client


def build_4band_payload(width: int = 64, height: int = 64, is_16bit: bool = False):
    """Helper to generate a valid 4-band upload dictionary for /api/predict."""
    if is_16bit:
        r_b = make_16bit_tiff_bytes(width, height)
        g_b = make_16bit_tiff_bytes(width, height)
        b_b = make_16bit_tiff_bytes(width, height)
        n_b = make_16bit_tiff_bytes(width, height)
        ext = "tif"
        mime = "image/tiff"
    else:
        r_b = make_png_bytes(width, height, color=100)
        g_b = make_png_bytes(width, height, color=110)
        b_b = make_png_bytes(width, height, color=120)
        n_b = make_png_bytes(width, height, color=130)
        ext = "png"
        mime = "image/png"

    return {
        "red": (f"red.{ext}", r_b, mime),
        "green": (f"green.{ext}", g_b, mime),
        "blue": (f"blue.{ext}", b_b, mime),
        "nir": (f"nir.{ext}", n_b, mime),
    }
