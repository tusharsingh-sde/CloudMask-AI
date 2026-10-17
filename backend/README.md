# CloudMask AI - Deep Learning Backend

Production-ready FastAPI service for multispectral satellite cloud detection and segmentation using a custom PyTorch U-Net architecture.

## 🚀 Features

- **PyTorch U-Net Model**: 4-channel input (Red, Green, Blue, NIR) -> 1-channel cloud mask output.
- **Low-Latency In-Memory Processing**: Streams and decodes satellite bands directly in memory without disk writing.
- **Security & DoS Protection**: Strict per-file size limits (max 25MB), file type verification, decompress bomb checks, and error sanitization.
- **Multi-Depth Image Normalization**: Handles 8-bit and 16-bit GeoTIFF / TIFF / PNG / JPG formats.
- **Rich Visual Output**: Encodes normalized RGB composite, binary mask, mask overlay, and Magma probability heatmaps to base64 Data URLs.

## 🛠️ Setup & Running

1. Activate your virtual environment:
   ```bash
   # Windows:
   ..\.venv\Scripts\activate
   # Linux / macOS:
   source ../.venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model checkpoint (if not already downloaded):
   ```bash
   python download_model.py
   ```

4. Start the API server:
   ```bash
   python api.py
   # Or using uvicorn directly:
   uvicorn api:app --host 127.0.0.1 --port 8000 --reload
   ```

## 📡 Endpoints

- `GET /api/health` - Server & model status, device info, and checkpoint loss.
- `POST /api/predict` - Accepts 4 bands (`red`, `green`, `blue`, `nir`) + optional `threshold` & `overlay_opacity`. Returns stats, RGB composite, mask, overlay, and heatmap.
- `POST /api/process` - Alias for `/api/predict`.
- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`
