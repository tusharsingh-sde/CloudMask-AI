<div align="center">

# ☁️ CloudMask AI

**Automated Multispectral Satellite Cloud Detection & Segmentation with Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Three.js](https://img.shields.io/badge/Three.js-0.186-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Pixel-level cloud segmentation in Landsat 8 multispectral satellite imagery using a hardened PyTorch U-Net architecture trained on the 38-Cloud dataset.*

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Frontend](#-frontend-workbench) · [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [API Reference](#-api-reference)
- [Frontend Workbench](#-frontend-workbench)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

CloudMask AI is an end-to-end deep learning platform for **automated cloud detection and segmentation** in multispectral Earth observation imagery. Accurate cloud detection is an indispensable preprocessing phase in satellite remote sensing — clouds obscure terrestrial targets and distort downstream spectral analyses such as crop yield assessment, wildfire tracking, deforestation monitoring, and flood mapping.

This system provides:
- A **hardened PyTorch U-Net model** delivering pixel-accurate cloud masks from 4 spectral bands: Red (B04), Green (B03), Blue (B02), and Near-Infrared (B05).
- A **production-grade FastAPI backend** engineered with sliding-window tiled inference, radiometric 16-bit preservation, pre-decode decompression bomb protection, in-memory sliding-window rate limiting, and safe checkpoint deserialization (`weights_only=True`).
- A **SpaceX/Planet Labs-inspired React 19 + Three.js Ground Station Workbench** featuring a 3D Earth WebGL hero simulation, multi-band batch dropzone, split-screen curtain slider, and 4-channel inspection.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Hardened U-Net Architecture** | Pure PyTorch encoder-decoder network with skip connections and bilinear interpolation fallback |
| 🛰️ **Radiometric Scaling Integrity** | Preserves 16-bit linear scaling (`/ 65535.0 * 255.0`), eliminating false-positive cloud masks on dark clear-sky scenes |
| 🛡️ **Pre-Decode Decompression Bomb Guard** | Inspects file headers before allocating RAM; rejects corrupt or oversized inputs ($> 4096 \times 4096$) |
| 🧩 **Sliding-Window Tiled Inference** | $384 \times 384$ sliding tiles with 2D feathered cosine blending for large satellite scenes, eliminating OOM crashes |
| ⚡ **Async Event-Loop Offloading** | Concurrent upload ingestion on the event loop with CPU/GPU inference offloaded to worker threads via `asyncio.to_thread` |
| 🔒 **Safe Model Lifespan** | Checkpoints loaded via FastAPI `lifespan` using `weights_only=True`, completely eliminating pickle code execution risks |
| ⏱️ **Sliding-Window Rate Limiting** | Zero-dependency per-IP rate limiter returning HTTP `429` with `Retry-After` headers and bounded memory |
| 🌐 **Container-Ready Probes** | Dedicated `/api/health` (liveness) and `/api/ready` (readiness) endpoints for Kubernetes, Cloud Run, and Docker |
| 🎛️ **Earth Observation Studio** | Batch drag-and-drop, vertical split comparison curtain slider, 4-pane synchronizer, and high-res PNG export |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CloudMask AI Platform                           │
├─────────────────────┬──────────────────────┬───────────────────────────┤
│  Frontend Studio    │   FastAPI Backend    │     ML Pipeline           │
│  React 19 + Three   │   Asynchronous API   │                           │
│  Three.js WebGL     │                      │  Dataset → Train → Model  │
│  Batch Dropzone     │  /api/predict        │                           │
│  Split Compare      │  /api/ready          │  38-Cloud    Pure PyTorch │
│  4-Pane Matrix      │  /api/health         │  4-Band      AMP (FP16)   │
│  Tabular Telemetry  │  Rate Limiting / CORS│  (R,G,B,NIR) Checkpoint   │
├─────────────────────┴──────────────────────┼───────────────────────────┤
│  Browser Client (Zero Invisible Elements)  │  Tiled Inference Engine   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Deep Learning** | PyTorch 2.x (pure PyTorch, zero `torchvision` runtime dependency) |
| **Computer Vision** | OpenCV (Headless for Docker/Cloud), Pillow, Tifffile |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, Python-Multipart |
| **Frontend UI** | React 19, Three.js, Lucide Icons, Plus Jakarta Sans, JetBrains Mono |
| **Tooling & Build** | Vite 8, Oxlint |
| **Training Pipeline** | CUDA, PyTorch AMP (Mixed Precision), CosineAnnealingLR, DataParallel |
| **Dataset** | [38-Cloud Landsat 8 Dataset](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ & npm
- CUDA-capable GPU *(recommended for training; CPU is fully supported for inference)*

### 1. Backend Setup

```bash
# 1. Clone repository
git clone https://github.com/tusharsingh-sde/CloudMask-AI.git
cd CloudMask-AI

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install production dependencies
cd backend
pip install -r requirements.txt

# 4. Download pre-trained U-Net weights (from GitHub Releases)
python download_model.py --skip-existing

# 5. Start API server
python api.py
# Server runs on http://127.0.0.1:8000 (Swagger docs at /docs)
```

### 2. Frontend Setup

In a separate terminal:

```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
# Studio Workbench opens on http://localhost:5173
```

---

## 🏋️ Training Pipeline (`backend/train.py`)

Train or fine-tune the U-Net on the 38-Cloud dataset:

```bash
cd backend
python train.py --epochs 20 --batch-size 32 --lr 3e-4 --val-split 0.15
```

### Supported CLI Flags

- `--data-dir PATH`: Directory containing `train_red`, `train_green`, `train_blue`, `train_nir`, `train_gt` (auto-detected).
- `--epochs INT`: Number of training epochs (default: `20`).
- `--batch-size INT`: Batch size (default: `32` on CUDA, `4` on CPU).
- `--lr FLOAT`: AdamW learning rate (default: `3e-4`) with `CosineAnnealingLR` scheduler.
- `--val-split FLOAT`: Validation set fraction (default: `0.15`, seeded deterministically).
- `--seed INT`: Random seed (default: `42`).
- `--resume PATH`: Resume training from an existing checkpoint (`cloudmask_unet_latest.pt`).
- `--save-every INT`: Save periodic checkpoint every N epochs (default: `0` = disabled to avoid disk bloat).

---

## 📡 API Reference & Status Codes

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness probe returning engine device and model status |
| `GET` | `/api/ready` | Readiness probe returning `200` when model is loaded, `503` when uninitialized |
| `POST` | `/api/predict` | Multispectral cloud segmentation pipeline |

### HTTP Status Codes

- **`200 OK`**: Inference completed successfully.
- **`400 Bad Request`**: Corrupt raster file, missing required band, invalid dimensions ($< 16\text{ px}$ or $> 4096\text{ px}$).
- **`413 Payload Too Large`**: Any individual band file exceeds $25\text{ MB}$.
- **`422 Unprocessable Entity`**: Parameter validation failure (`threshold` outside $0.05..0.95$ or `overlay_opacity` outside $0.1..1.0$).
- **`429 Too Many Requests`**: Rate limit exceeded (includes standard `Retry-After` header).
- **`503 Service Unavailable`**: Model weights are not loaded.

---

## 🖥 Frontend Workbench

The React 19 Ground Station provides:
- **Cinematic WebGL Space Simulation**: Rotating Earth with layered atmospheric glow, 3D undulating particle wave, and micro-satellites.
- **Smart Multi-Band Ingestion**: Drag & drop all 4 band files simultaneously; automatically identifies Red (B04), Green (B03), Blue (B02), and NIR (B05) from filenames.
- **Split Comparison Curtain Slider**: Interactive comparison between True-Color RGB and segmented cloud masks with mouse drag, touch, and arrow-key navigation.
- **Multi-View Modes**: Split View, Single Channel, and 2×2 Synchronized Quad Matrix.
- **High-Resolution Exports**: Download individual channels or batch export all 4 imagery layers as PNGs.

---

## 📁 Project Structure

```
CloudMask-AI/
│
├── backend/
│   ├── api.py                 # Hardened FastAPI service with tiled inference & rate limiting
│   ├── model.py               # Pure PyTorch U-Net architecture definition
│   ├── dataset.py             # 38-Cloud dataset loader (pure PyTorch tensor conversion)
│   ├── train.py               # Training loop with validation split, AMP, and metrics
│   ├── download_model.py      # Atomic model downloader with checksum & CI flags
│   ├── requirements.txt       # Lean production inference dependencies
│   ├── requirements-train.txt # Model training dependencies
│   ├── requirements-dev.txt   # Automated test dependencies
│   ├── .env.example           # Documented environment variable template
│   └── README.md              # Backend documentation
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── EarthScene.jsx       # Three.js 3D Earth, particle wave & satellites
│   │   │   ├── HeroSection.jsx      # Minimalist editorial aerospace hero
│   │   │   ├── Navbar.jsx           # Glass sticky navbar with live telemetry
│   │   │   └── StudioWorkbench.jsx  # Multi-band dropzone, split slider & quad matrix
│   │   ├── constants/               # Landsat 8 band definitions (B02..B05)
│   │   ├── styles/tokens.css        # Deep space design system tokens
│   │   ├── App.jsx                  # Application state & section tracking
│   │   └── App.css                  # Tailored aerospace ground station styling
│   ├── index.html
│   ├── package.json
│   └── vite.config.js               # Reverse proxy configuration
│
└── README.md                        # Master project documentation
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- **38-Cloud Dataset** by Sorour Mohajerani and Parvaneh Saeedi ([Paper](https://arxiv.org/abs/1901.03358))
- **U-Net Architecture** by Ronneberger et al. ([Paper](https://arxiv.org/abs/1505.04597))
- **USGS / NASA** for Landsat 8 multispectral Earth observation imagery
