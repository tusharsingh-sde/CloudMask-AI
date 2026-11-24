import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional, Tuple

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tifffile
import torch
import torchvision.transforms.functional as TF
import uvicorn

from model import UNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CloudMaskAI")

app = FastAPI(
    title="CloudMask AI API",
    description="Deep Learning API for Multispectral Satellite Cloud Detection and Segmentation",
    version="1.0.0"
)

# Security: CORS configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Constants for security, bounds & concurrency
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per band
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
MAX_DIMENSION = 4096
DEFAULT_TILE_SIZE = 384
DEFAULT_OVERLAP = 64

# Concurrency bound: prevents multiple heavy inferences from causing OOM
inference_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCE", "2")))

# Setup Device & Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Setting up PyTorch device: {device}...")

model = UNet(in_channels=4, out_channels=1).to(device)
model.eval()  # Unconditionally set eval mode

def resolve_model_path() -> str:
    candidates = [
        os.getenv("MODEL_PATH"),
        "cloudmask_unet_best.pt",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudmask_unet_best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cloudmask_unet_best.pt"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudmask_unet_best.pt")

model_path = resolve_model_path()

model_loaded: bool = False
checkpoint_meta: dict = {}

if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model_loaded = True
        checkpoint_meta = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "best_loss": float(checkpoint.get("best_loss", 0.0))
        }
        logger.info(f"✅ Model loaded successfully! (Epoch: {checkpoint_meta['epoch']}, Loss: {checkpoint_meta['best_loss']:.4f})")
    except Exception as e:
        model_loaded = False
        logger.error(f"❌ Failed to load model weights from {model_path}: {e}", exc_info=True)
else:
    model_loaded = False
    logger.warning(f"⚠️ Model checkpoint '{model_path}' not found! Inference will return HTTP 503 until provided.")


def encode_image_to_base64(img_array: np.ndarray, format: str = ".png") -> str:
    """Encode OpenCV/numpy image array to base64 Data URL."""
    success, buffer = cv2.imencode(format, img_array)
    if not success:
        raise ValueError("Failed to encode image to buffer.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if format == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def validate_image_header(contents: bytes, filename: str) -> Tuple[int, int]:
    """
    Inspect (width, height) from file headers BEFORE decompressing pixel data into RAM.
    Guards against decompression bombs and invalid dimensions.
    Returns (width, height).
    Raises ValueError if invalid, out of bounds, or corrupted.
    """
    ext = os.path.splitext(filename)[1].lower()
    width, height = 0, 0
    
    if ext in {".tif", ".tiff"}:
        try:
            with io.BytesIO(contents) as bio:
                with tifffile.TiffFile(bio) as tif:
                    if not tif.pages:
                        raise ValueError(f"TIFF file '{filename}' contains no valid pages.")
                    page = tif.pages[0]
                    shape = page.shape
                    if len(shape) == 2:
                        height, width = int(shape[0]), int(shape[1])
                    elif len(shape) >= 3:
                        height, width = int(shape[0]), int(shape[1])
                    else:
                        raise ValueError(f"TIFF file '{filename}' has invalid shape {shape}.")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse TIFF header for '{filename}': {e}")
            
    elif ext in {".png", ".jpg", ".jpeg"}:
        try:
            # Enforce max pixel limit on PIL to prevent memory exhaustion
            Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION
            with io.BytesIO(contents) as bio:
                with Image.open(bio) as img:
                    width, height = img.size  # PIL reports (width, height)
        except Exception as e:
            raise ValueError(f"Failed to parse image header for '{filename}': {e}")
    else:
        raise ValueError(
            f"Band file '{filename}' has unsupported extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    if width < 16 or height < 16:
        raise ValueError(f"Image '{filename}' resolution ({width}x{height}) is too small (minimum 16x16).")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError(f"Image '{filename}' resolution ({width}x{height}) exceeds maximum allowed ({MAX_DIMENSION}x{MAX_DIMENSION}).")
    if (width * height) > (MAX_DIMENSION * MAX_DIMENSION):
        raise ValueError(f"Image '{filename}' pixel count ({width * height}) exceeds maximum allowed ({MAX_DIMENSION * MAX_DIMENSION}).")

    return width, height


def decode_band_image(contents: bytes, filename: str) -> np.ndarray:
    """
    Safely decode uploaded image file bytes to a single-channel 2D uint8 numpy array.
    Supports 8-bit and 16-bit satellite GeoTIFF, TIFF, PNG, and JPEG.
    Preserves training-consistent radiometric scaling without per-image min-max distortion.
    """
    # 1. Header validation before decompressing pixel buffer
    validate_image_header(contents, filename)

    # 2. Try decoding with OpenCV preserving bit depth
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # Fallback to tifffile if OpenCV couldn't parse complex TIFF
    if img is None:
        try:
            with io.BytesIO(contents) as bio:
                img = tifffile.imread(bio)
        except Exception as e:
            logger.debug(f"tifffile fallback failed for {filename}: {e}")
            
    if img is None:
        raise ValueError(f"File '{filename}' is corrupt or an unsupported image format.")
        
    # If 3D, extract first channel
    if img.ndim == 3:
        img = img[:, :, 0]
    elif img.ndim != 2:
        raise ValueError(f"Image '{filename}' has invalid dimensions: shape {img.shape}.")
        
    h, w = img.shape
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValueError(f"Image '{filename}' resolution ({w}x{h}) exceeds maximum allowed ({MAX_DIMENSION}x{MAX_DIMENSION}).")
    if h < 16 or w < 16:
        raise ValueError(f"Image '{filename}' resolution ({w}x{h}) is too small (minimum 16x16).")
        
    # Radiometric scaling consistent with training dataset (cv2.imread(..., 0) -> uint8)
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        # Match cv2.imread(..., 0) linear 16-bit-to-8-bit scaling (/ 256.0 or / 65535.0 * 255.0)
        return np.clip(img.astype(np.float32) / 65535.0 * 255.0, 0, 255).astype(np.uint8)
    elif np.issubdtype(img.dtype, np.floating):
        img_float = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        max_val = float(np.max(img_float)) if img_float.size > 0 else 0.0
        if max_val <= 1.0:
            img_float = img_float * 255.0
        return np.clip(img_float, 0, 255).astype(np.uint8)
    else:
        # Other integer types (e.g. int16, int32)
        img_clipped = np.clip(img, 0, None).astype(np.float32)
        max_val = float(np.max(img_clipped)) if img_clipped.size > 0 else 0.0
        if max_val > 65535.0:
            return np.clip(img_clipped / max_val * 255.0, 0, 255).astype(np.uint8)
        elif max_val > 255.0:
            return np.clip(img_clipped / 65535.0 * 255.0, 0, 255).astype(np.uint8)
        else:
            return np.clip(img_clipped, 0, 255).astype(np.uint8)


def create_rgb_composite(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Generate normalized RGB composite for visualization.
    Applies gentle 2nd–98th percentile stretch across the combined RGB stack with fallback
    so the user's RGB preview looks crisp without corrupting the U-Net input tensor.
    """
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255)
    else:
        rgb_min, rgb_max = float(rgb.min()), float(rgb.max())
        if rgb_max > rgb_min:
            rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255.0)
        else:
            rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def create_overlay(rgb: np.ndarray, mask: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    """Overlay predicted cloud mask (orbital cyan) onto the RGB composite."""
    overlay = rgb.copy().astype(np.float32)
    # Orbital cyan [56, 189, 248] matching deep space aerospace UI
    cloud_color = np.array([56, 189, 248], dtype=np.float32)
    cloud_mask_3ch = np.stack([mask, mask, mask], axis=-1)
    
    overlay = np.where(
        cloud_mask_3ch > 0,
        overlay * (1.0 - opacity) + cloud_color * opacity,
        overlay
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_tapered_window(tile_size: int, overlap: int) -> np.ndarray:
    """Create a 2D cosine/feathered weight window to smoothly blend overlapping tiles."""
    w1d = np.ones(tile_size, dtype=np.float32)
    if overlap > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32)))
        ramp = np.clip(ramp, 0.05, 1.0)
        w1d[:overlap] = ramp
        w1d[-overlap:] = ramp[::-1]
    w2d = np.outer(w1d, w1d)
    return w2d.astype(np.float32)


def run_tiled_inference(
    img_stacked: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> Tuple[np.ndarray, bool]:
    """
    Execute U-Net inference on 4-channel image (H, W, 4) using sliding-window tiling.
    If H <= tile_size and W <= tile_size, runs a single pass with reflect padding to multiple of 16.
    If larger than tile_size, uses sliding-window tiles with cosine feathered blending to avoid seams.
    Returns (probabilities_map, is_tiled).
    """
    h, w, c = img_stacked.shape
    if c != 4:
        raise ValueError(f"Expected 4 channels for inference, got shape {img_stacked.shape}")
        
    is_tiled = h > tile_size or w > tile_size
    
    if not is_tiled:
        # Single tile pass with reflect padding to multiple of 16 for U-Net down/up pooling
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(img_stacked, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            padded = img_stacked
            
        tensor = TF.to_tensor(padded).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(tensor)
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            
        if probs.ndim == 0:
            probs = np.array([[probs.item()]], dtype=np.float32)
            
        if pad_h > 0 or pad_w > 0:
            probs = probs[:h, :w]
            
        return probs.astype(np.float32), False
        
    # Sliding-window inference across large scenes
    stride = max(1, tile_size - overlap)
    
    # If h or w is smaller than tile_size, pad to tile_size first
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h > 0 or pad_w > 0:
        working_img = np.pad(img_stacked, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        working_img = img_stacked
        
    work_h, work_w, _ = working_img.shape
    
    def get_starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size, stride))
        last_start = length - tile_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    y_starts = get_starts(work_h)
    x_starts = get_starts(work_w)
    
    prob_accum = np.zeros((work_h, work_w), dtype=np.float32)
    weight_accum = np.zeros((work_h, work_w), dtype=np.float32)
    window = create_tapered_window(tile_size, overlap)
    
    for y in y_starts:
        for x in x_starts:
            tile = working_img[y : y + tile_size, x : x + tile_size]
            tile_tensor = TF.to_tensor(tile).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                tile_out = model(tile_tensor)
                tile_prob = torch.sigmoid(tile_out).squeeze().cpu().numpy()
                
            if tile_prob.ndim == 0:
                tile_prob = np.array([[tile_prob.item()]], dtype=np.float32)
                
            prob_accum[y : y + tile_size, x : x + tile_size] += tile_prob * window
            weight_accum[y : y + tile_size, x : x + tile_size] += window

    # Normalize by accumulated feathered weights
    final_probs = prob_accum / np.maximum(weight_accum, 1e-7)
    
    # Crop back to original (h, w)
    if pad_h > 0 or pad_w > 0:
        final_probs = final_probs[:h, :w]
        
    return final_probs.astype(np.float32), True


def _run_inference_pipeline(
    r_bytes: bytes, r_fname: str,
    g_bytes: bytes, g_fname: str,
    b_bytes: bytes, b_fname: str,
    n_bytes: bytes, n_fname: str,
    threshold: float,
    overlay_opacity: float,
    start_time: float
) -> dict:
    """
    Synchronous CPU/GPU heavy inference pipeline.
    Runs off the main asyncio event loop via asyncio.to_thread().
    """
    # 1. Decode all 4 bands with radiometric integrity & decompression bomb guard
    r = decode_band_image(r_bytes, r_fname)
    g = decode_band_image(g_bytes, g_fname)
    b = decode_band_image(b_bytes, b_fname)
    n = decode_band_image(n_bytes, n_fname)
    
    # 2. Verify and harmonize dimensions to red band
    target_shape = r.shape
    target_h, target_w = target_shape[0], target_shape[1]
    if g.shape != target_shape or b.shape != target_shape or n.shape != target_shape:
        logger.info(f"Band dimension mismatch detected. Harmonizing to {target_w}x{target_h}.")
        if g.shape != target_shape:
            g = cv2.resize(g, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if b.shape != target_shape:
            b = cv2.resize(b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if n.shape != target_shape:
            n = cv2.resize(n, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 3. Stack into (H, W, 4) uint8 array
    img_stacked = np.stack([r, g, b, n], axis=-1)
    
    # 4. Neural Network Inference (Tiled / Single Pass)
    probabilities, is_tiled = run_tiled_inference(
        img_stacked,
        tile_size=DEFAULT_TILE_SIZE,
        overlap=DEFAULT_OVERLAP
    )
    predicted_mask = (probabilities > threshold).astype(np.float32)
    
    # 5. Calculate Geospatial Coverage Metrics
    total_pixels = int(predicted_mask.size)
    cloud_pixels = int(np.count_nonzero(predicted_mask))
    clear_pixels = int(total_pixels - cloud_pixels)
    cloud_percentage = round((cloud_pixels / total_pixels) * 100.0, 2)
    clear_percentage = round(100.0 - cloud_percentage, 2)
    
    # 6. Generate Visual Imagery & Heatmap
    rgb_composite = create_rgb_composite(r, g, b)
    mask_uint8 = (predicted_mask * 255).astype(np.uint8)
    overlay_img = create_overlay(rgb_composite, predicted_mask, opacity=overlay_opacity)
    
    # Magma heatmap of cloud probabilities
    prob_heatmap = cv2.applyColorMap((probabilities * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    prob_heatmap = cv2.cvtColor(prob_heatmap, cv2.COLOR_BGR2RGB)
    
    # 7. Convert images to Base64 Data URLs
    rgb_url = encode_image_to_base64(cv2.cvtColor(rgb_composite, cv2.COLOR_RGB2BGR))
    mask_url = encode_image_to_base64(mask_uint8)
    overlay_url = encode_image_to_base64(cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    heatmap_url = encode_image_to_base64(cv2.cvtColor(prob_heatmap, cv2.COLOR_RGB2BGR))
    
    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    
    metadata = {
        "dimensions": f"{target_w} x {target_h} px",
        "threshold": threshold,
        "overlay_opacity": overlay_opacity,
        "device": device,
        "inference_time_ms": elapsed_ms,
        "bands_analyzed": ["Red", "Green", "Blue", "NIR"],
        "model_epoch": checkpoint_meta.get("epoch", "N/A"),
        "model_best_loss": checkpoint_meta.get("best_loss", "N/A"),
        "tiled_inference": is_tiled,
        "tile_size": DEFAULT_TILE_SIZE
    }
    
    return {
        "status": "success",
        "cloud_coverage_percentage": cloud_percentage,
        "clear_sky_percentage": clear_percentage,
        "cloud_pixels": cloud_pixels,
        "clear_pixels": clear_pixels,
        "total_pixels": total_pixels,
        # Both formats for maximum frontend & backward compatibility
        "imageUrl": overlay_url,
        "rgb_composite_url": rgb_url,
        "mask_url": mask_url,
        "overlay_url": overlay_url,
        "heatmap_url": heatmap_url,
        "metadata": metadata,
        "message": f"Cloud detection completed with {cloud_percentage}% cloud coverage."
    }


async def read_uploaded_file_bytes(file: UploadFile, band_name: str) -> Tuple[str, bytes]:
    """Read upload stream with size limit and file type verification."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail=f"Band '{band_name}' is missing or has no filename.")
        
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Band '{band_name}' has invalid extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
        
    # Read with size cap
    contents = await file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Band '{band_name}' exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail=f"Band '{band_name}' file is empty.")
        
    return file.filename, contents


@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CloudMask AI Backend",
        "device": device,
        "model_loaded": model_loaded,
        "checkpoint": checkpoint_meta
    }


async def execute_prediction(
    red: UploadFile,
    green: UploadFile,
    blue: UploadFile,
    nir: UploadFile,
    threshold: float = 0.5,
    overlay_opacity: float = 0.45
):
    # Requirement 2: Block inference immediately when model weights are not loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. Run download_model.py or set MODEL_PATH."
        )

    start_time = time.time()
    
    # Clamp parameters safely
    threshold = max(0.05, min(0.95, float(threshold)))
    overlay_opacity = max(0.1, min(1.0, float(overlay_opacity)))
    
    # 1. Read all 4 uploaded files concurrently on the event loop
    (r_fname, r_bytes), (g_fname, g_bytes), (b_fname, b_bytes), (n_fname, n_bytes) = await asyncio.gather(
        read_uploaded_file_bytes(red, "red"),
        read_uploaded_file_bytes(green, "green"),
        read_uploaded_file_bytes(blue, "blue"),
        read_uploaded_file_bytes(nir, "nir")
    )
    
    # 2. Bound concurrency with semaphore and offload CPU/GPU work from event loop
    async with inference_semaphore:
        result = await asyncio.to_thread(
            _run_inference_pipeline,
            r_bytes, r_fname,
            g_bytes, g_fname,
            b_bytes, b_fname,
            n_bytes, n_fname,
            threshold,
            overlay_opacity,
            start_time
        )
        
    return result


# Route handlers for all common endpoints
@app.post("/predict")
@app.post("/api/predict")
@app.post("/api/process")
async def predict_endpoint(
    red: UploadFile = File(...),
    green: UploadFile = File(...),
    blue: UploadFile = File(...),
    nir: UploadFile = File(...),
    threshold: Optional[float] = Form(0.5),
    overlay_opacity: Optional[float] = Form(0.45)
):
    try:
        return await execute_prediction(
            red=red,
            green=green,
            blue=blue,
            nir=nir,
            threshold=threshold if threshold is not None else 0.5,
            overlay_opacity=overlay_opacity if overlay_opacity is not None else 0.45
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in input image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference pipeline error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during satellite image processing. Please verify file integrity."}
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
