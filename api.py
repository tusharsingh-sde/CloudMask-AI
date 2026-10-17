from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import io
import os
import time
import base64
import logging
from typing import Optional

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

# Constants for security and bounds
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per band
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
MAX_DIMENSION = 4096

# Setup Device & Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Setting up PyTorch device: {device}...")

model = UNet(in_channels=4, out_channels=1).to(device)
model_path = os.getenv("MODEL_PATH", "cloudmask_unet_best.pt")

checkpoint_meta = {}
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        checkpoint_meta = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "best_loss": float(checkpoint.get("best_loss", 0.0))
        }
        logger.info(f"✅ Model loaded successfully! (Epoch: {checkpoint_meta['epoch']}, Loss: {checkpoint_meta['best_loss']:.4f})")
    except Exception as e:
        logger.error(f"❌ Failed to load model weights: {e}", exc_info=True)
else:
    logger.warning(f"⚠️ Model checkpoint {model_path} not found! Predictions will fail until provided.")


def encode_image_to_base64(img_array: np.ndarray, format: str = ".png") -> str:
    """Encode OpenCV/numpy image array to base64 Data URL."""
    success, buffer = cv2.imencode(format, img_array)
    if not success:
        raise ValueError("Failed to encode image to buffer.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if format == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def decode_band_image(contents: bytes, filename: str) -> np.ndarray:
    """
    Safely decode uploaded image file bytes to a single-channel 2D uint8 numpy array.
    Supports 8-bit and 16-bit satellite GeoTIFF, TIFF, PNG, and JPEG.
    """
    nparr = np.frombuffer(contents, np.uint8)
    
    # 1. Try decoding with OpenCV preserving bit depth
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # 2. Fallback to tifffile if OpenCV couldn't parse complex TIFF
    if img is None:
        try:
            import tifffile
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
        
    # Security: check dimensions to avoid decompress bombs
    h, w = img.shape
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValueError(f"Image '{filename}' resolution ({w}x{h}) exceeds maximum allowed ({MAX_DIMENSION}x{MAX_DIMENSION}).")
    if h < 16 or w < 16:
        raise ValueError(f"Image '{filename}' resolution ({w}x{h}) is too small (minimum 16x16).")
        
    # Normalize to uint8 (0-255)
    if img.dtype != np.uint8:
        img_float = img.astype(np.float32)
        img_min = float(np.min(img_float))
        img_max = float(np.max(img_float))
        if img_max > img_min:
            img = ((img_float - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
            
    return img


async def read_and_validate_file(file: UploadFile, band_name: str) -> np.ndarray:
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
        
    try:
        return decode_band_image(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def create_rgb_composite(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generate normalized RGB composite for visualization."""
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb_min, rgb_max = float(rgb.min()), float(rgb.max())
    if rgb_max > rgb_min:
        rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255.0).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def create_overlay(rgb: np.ndarray, mask: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    """Overlay predicted cloud mask (neon magenta/cyan) onto the RGB composite."""
    overlay = rgb.copy().astype(np.float32)
    # Bright high-visibility neon magenta for satellite cloud detection
    cloud_color = np.array([255, 30, 140], dtype=np.float32)
    cloud_mask_3ch = np.stack([mask, mask, mask], axis=-1)
    
    overlay = np.where(
        cloud_mask_3ch > 0,
        overlay * (1.0 - opacity) + cloud_color * opacity,
        overlay
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CloudMask AI Backend",
        "device": device,
        "model_loaded": checkpoint_meta.get("epoch") is not None,
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
    start_time = time.time()
    
    # Clamp parameters safely
    threshold = max(0.05, min(0.95, float(threshold)))
    overlay_opacity = max(0.1, min(1.0, float(overlay_opacity)))
    
    # 1. Read and validate all 4 bands concurrently
    r = await read_and_validate_file(red, "red")
    g = await read_and_validate_file(green, "green")
    b = await read_and_validate_file(blue, "blue")
    n = await read_and_validate_file(nir, "nir")
    
    # 2. Verify and align dimensions
    target_shape = r.shape
    if g.shape != target_shape or b.shape != target_shape or n.shape != target_shape:
        # Resize to match red band's (width, height)
        target_w, target_h = target_shape[1], target_shape[0]
        logger.info(f"Band dimension mismatch detected. Harmonizing to {target_w}x{target_h}.")
        if g.shape != target_shape:
            g = cv2.resize(g, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if b.shape != target_shape:
            b = cv2.resize(b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if n.shape != target_shape:
            n = cv2.resize(n, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
    # 3. Stack into (1, 4, H, W) tensor
    img_stacked = np.stack([r, g, b, n], axis=-1)
    image_tensor = TF.to_tensor(img_stacked).unsqueeze(0).to(device)
    
    # 4. Neural Network Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
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
    
    # Format metadata & return rich payload
    metadata = {
        "dimensions": f"{target_shape[1]} x {target_shape[0]} px",
        "threshold": threshold,
        "overlay_opacity": overlay_opacity,
        "device": device,
        "inference_time_ms": elapsed_ms,
        "bands_analyzed": ["Red", "Green", "Blue", "NIR"],
        "model_epoch": checkpoint_meta.get("epoch", "N/A"),
        "model_best_loss": checkpoint_meta.get("best_loss", "N/A")
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
    except Exception as e:
        logger.error(f"Inference pipeline error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during satellite image processing. Please verify file integrity."}
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)