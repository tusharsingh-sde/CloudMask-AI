# CloudMask AI - Deep Learning Backend

Production-grade FastAPI service and PyTorch ML pipeline for multispectral satellite cloud detection and segmentation on Landsat 8 imagery.

---

## 🚀 Key Architectural Features

- **Hardened PyTorch U-Net**: 4-channel input (Red, Green, Blue, NIR) $\to$ 1-channel raw logits output. Pure PyTorch implementation with zero `torchvision` runtime dependency.
- **Radiometric Scaling Integrity**: Preserves linear 16-bit to 8-bit scaling matching training data (`cv2.imread(..., 0)`), preventing false-positive cloud masks on dark clear-sky scenes.
- **Pre-Decode Decompression Bomb Guard**: Validates image headers (`width`, `height`, pixel count $\le 4096 \times 4096$) before decompressing pixel buffers into RAM.
- **Sliding-Window Tiled Inference**: Seamless $384 \times 384$ tiled inference with cosine feathered blending for large satellite scenes ($1024 \times 1024$, $2048 \times 2048$, $4096 \times 4096$), eliminating Out-Of-Memory (OOM) crashes.
- **Async Event-Loop Offloading**: Concurrently streams uploads on the event loop, offloading CPU/GPU-bound inference to worker threads via `asyncio.to_thread` with bounded concurrency (`MAX_CONCURRENT_INFERENCE`).
- **Safe Model Lifespan**: Checkpoints are loaded under FastAPI `lifespan` using `weights_only=True` (safe from pickle code execution).
- **Security & Rate Limiting**: Built-in sliding-window rate limiting (`429 Too Many Requests` with `Retry-After`), strict CORS without wildcard credentialing, and security headers (`nosniff`, `DENY`, `no-store`).

---

## 🛠️ Setup & Running

### 1. Environment & Dependencies

```bash
# Navigate to backend directory
cd backend

# Install production inference dependencies
pip install -r requirements.txt

# For training or development:
# pip install -r requirements-train.txt
# pip install -r requirements-dev.txt
```

### 2. Download Model Checkpoint

```bash
python download_model.py
```

Options:
- `--skip-existing`: Exits cleanly with code 0 if `cloudmask_unet_best.pt` already exists (ideal for Docker / CI builds).
- `--force`: Overwrites existing checkpoint without interactive confirmation.
- `--output PATH`: Specify custom output destination path.
- `--tag TAG`: Download from a specific GitHub Release tag instead of latest.

### 3. Start API Server

```bash
python api.py
# Or with uvicorn:
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 📡 API Reference & HTTP Status Codes

### Endpoints

| Method | Endpoint | Description | Response Model |
|---|---|---|---|
| `GET` | `/` / `/api/health` | Liveness probe returning engine & model status | `HealthResponse` |
| `GET` | `/api/ready` | Readiness probe for container orchestrators | `ReadyResponse` |
| `POST` | `/api/predict` | 4-band satellite cloud segmentation pipeline | `PredictionResponse` |

### HTTP Status Codes

- **`200 OK`**: Successful inference or health/readiness check.
- **`400 Bad Request`**: Corrupt raster file, missing required band, invalid dimension ($< 16$ px or $> 4096$ px), or unsupported file format.
- **`413 Payload Too Large`**: Any individual band file exceeds $25\text{ MB}$.
- **`422 Unprocessable Entity`**: Parameter validation failure (`threshold` outside $0.05..0.95$ or `overlay_opacity` outside $0.1..1.0$).
- **`429 Too Many Requests`**: Sliding-window rate limit exceeded. Includes standard `Retry-After` header.
- **`503 Service Unavailable`**: Model weights are missing or not loaded (`/api/ready` and `/api/predict`).

---

## ⚙️ Configuration (.env)

See `.env.example` for all configurable environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `cloudmask_unet_best.pt` | Custom checkpoint path |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `ALLOWED_ORIGINS` | `localhost:5173,3000` | CORS permitted origins |
| `TRUST_PROXY` | `false` | Enable only behind trusted reverse proxy |
| `MAX_CONCURRENT_INFERENCE` | `2` | Max parallel inference threads |
| `RATE_LIMIT_ENABLED` | `true` | In-memory sliding-window limiter |
| `RATE_LIMIT_REQUESTS` | `15` | Max requests per IP window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limiter window length |

---

## 🏋️ Model Training (`train.py`)

Train the U-Net model on the 38-Cloud dataset:

```bash
python train.py --epochs 20 --batch-size 32 --val-split 0.15 --lr 3e-4
```

### CLI Arguments

- `--data-dir PATH`: Directory containing `train_red`, `train_green`, `train_blue`, `train_nir`, `train_gt` (auto-detected).
- `--epochs INT`: Number of training epochs (default: `20`).
- `--batch-size INT`: Batch size (default: `32` on CUDA, `4` on CPU).
- `--lr FLOAT`: AdamW learning rate (default: `3e-4`) with `CosineAnnealingLR`.
- `--val-split FLOAT`: Deterministic validation split fraction (default: `0.15`, seed: `42`).
- `--resume PATH`: Resume training from an existing checkpoint (`cloudmask_unet_latest.pt`).
- `--save-every INT`: Save periodic checkpoint every N epochs (default: `0` = disabled to save disk space).
