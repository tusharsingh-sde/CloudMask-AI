<div align="center">

# ☁️ CloudMask AI

**Automated Cloud Segmentation for Satellite Imagery using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Three.js](https://img.shields.io/badge/Three.js-0.186-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Pixel-level cloud detection in Landsat 8 multispectral satellite imagery using a U-Net architecture trained on the 38-Cloud dataset.*

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Frontend](#-frontend-dashboard) · [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [API Reference](#-api-reference)
- [Frontend Dashboard](#-frontend-dashboard)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

CloudMask AI is an end-to-end deep learning pipeline for **automatic cloud detection** in satellite imagery. Accurate cloud masking is a critical preprocessing step in remote sensing — clouds obscure the Earth's surface and degrade the quality of downstream analyses like land cover classification, change detection, and vegetation monitoring.

This project provides:
- A **trained U-Net model** that segments clouds at pixel level from 4-band (Red, Green, Blue, NIR) Landsat 8 patches
- A **hardened FastAPI REST API** for low-latency in-memory inference with DoS protection
- A **modern React 19 + Three.js frontend** featuring real-time 3D Earth orbital simulation and multispectral analysis dashboard

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **U-Net Segmentation** | Custom encoder-decoder architecture with skip connections for precise pixel-level cloud masks |
| ⚡ **Mixed Precision Training** | FP16 training with `torch.amp` for 2x faster training and lower VRAM usage |
| 🔥 **Multi-GPU Support** | Automatic `DataParallel` scaling across multiple GPUs |
| 🌐 **Secure REST API** | FastAPI backend with reverse proxy support, strict CORS, and in-memory multi-depth image decoding |
| 🛰️ **3D Orbital WebGL Frontend** | Interactive Three.js space simulation with live satellites and responsive telemetry |
| 🖼️ **Multi-View Imagery** | True-Color RGB composite, binary mask, neon magenta overlay, and continuous Magma heatmaps |
| 📊 **Cloud Coverage Stats** | Quantitative cloud coverage %, clear sky %, pixel counts, and latency metrics |
| 💾 **Smart Checkpointing** | Per-epoch, latest, and best model saves with full resume support |
| 🔄 **Cross-Platform** | Seamless execution on Windows, Linux, and Cloud environments |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────┐
│                    CloudMask AI                        │
├──────────────┬──────────────┬──────────────────────────┤
│  Frontend    │   Backend    │     ML Pipeline          │
│  React 19    │   FastAPI    │                          │
│  Three.js    │   REST API   │  Dataset → Train → Model │
│  Vite        │              │                          │
│              │  /predict    │  38-Cloud   UNet   .pt   │
│  Dropzones   │  /process    │  Dataset    FP16   Ckpt  │
│  Telemetry   │  /health     │                          │
├──────────────┴──────────────┼──────────────────────────┤
│        Reverse Proxy / CORS │  Inference Engine        │
└─────────────────────────────┴──────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch, torchvision |
| **Computer Vision** | OpenCV, PIL, tifffile |
| **Backend API** | FastAPI, Uvicorn, python-multipart |
| **Frontend UI** | React 19, Three.js, Lucide Icons, Vanilla CSS |
| **Build Tool** | Vite, Oxlint |
| **Training Infra** | CUDA, AMP (Mixed Precision), DataParallel |
| **Dataset** | [38-Cloud (Landsat 8)](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ & npm
- CUDA-capable GPU *(recommended for training, CPU works for inference)*

### 1. Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/tusharsingh-sde/CloudMask-AI.git
cd CloudMask-AI

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Navigate to backend and install dependencies
cd backend
pip install -r requirements.txt

# 4. Download pre-trained checkpoint (if not already present)
python download_model.py

# 5. Start FastAPI Backend
python api.py
# Server runs on http://127.0.0.1:8000
```

### 2. Frontend Setup

In a new terminal window:

```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
# Dashboard opens on http://localhost:5173
```

---

## 🏋️ Training

### Dataset Setup

Download the **38-Cloud** dataset from Kaggle or GitHub:

```
38-Cloud_training/
├── train_red/       # Band 4 (Red)
├── train_green/     # Band 3 (Green)
├── train_blue/      # Band 2 (Blue)
├── train_nir/       # Band 5 (Near Infrared)
├── train_gt/        # Binary cloud masks (ground truth)
└── training_patches_38-Cloud.csv
```

### Run Training

```bash
# Default: 20 epochs, batch size 16, auto device selection
python train.py

# Custom parameters:
python train.py --epochs 30 --batch_size 32 --lr 0.0005
```

---

## 📡 API Reference

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "online",
  "service": "CloudMask AI Backend",
  "device": "cpu",
  "model_loaded": true,
  "checkpoint": {
    "epoch": 11,
    "best_loss": 0.0958
  }
}
```

### Run Prediction

```http
POST /api/predict
Content-Type: multipart/form-data
```

**Parameters:**
| Field | Type | Required | Description |
|---|---|---|---|
| `red` | File | Yes | Band 4 (Red) satellite image (`.tif`, `.png`, `.jpg`) |
| `green` | File | Yes | Band 3 (Green) satellite image |
| `blue` | File | Yes | Band 2 (Blue) satellite image |
| `nir` | File | Yes | Band 5 (NIR) satellite image |
| `threshold` | Float | No | Confidence threshold (default `0.5`) |
| `overlay_opacity` | Float | No | Mask opacity (default `0.45`) |

---

## 🖥 Frontend Dashboard

The frontend application provides:
- **3D Interactive Orbit Visualizer**: Real-time rotating Earth with orbital paths and satellites.
- **Multi-Band Dropzones**: Drag & drop support for all 4 spectral channels with file validation.
- **Dual Controls**: Confidence Threshold and Mask Overlay Opacity sliders.
- **4-Channel View**: Switch seamlessly between Mask Overlay, Binary Mask, RGB Composite, and Magma Heatmap.
- **Exporting**: One-click high-resolution PNG downloads for any generated visual product.

---

## 📁 Project Structure

```
CloudMask-AI/
│
├── backend/              # FastAPI Python service & PyTorch ML pipeline
│   ├── api.py            # Hardened REST API for inference
│   ├── model.py          # U-Net architecture definition
│   ├── dataset.py        # 38-Cloud dataset loader (4-band + ground truth)
│   ├── train.py          # Training loop (AMP, multi-GPU, checkpointing)
│   ├── download_model.py # Auto-download model checkpoint from GitHub Releases
│   ├── requirements.txt  # Python backend dependencies
│   └── README.md         # Backend documentation
│
├── frontend/             # Modern React 19 + Three.js web application
│   ├── src/
│   │   ├── components/   # EarthScene, LandingPage, ProcessingDashboard, ResultViewer, Sidebar
│   │   ├── constants/    # Satellite band definitions
│   │   ├── services/     # API client with health checks & timeouts
│   │   ├── App.jsx       # State & workflow coordinator
│   │   └── App.css       # Deep-space neon styling
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js    # Reverse proxy configuration
│   └── README.md         # Frontend documentation
│
├── .gitignore            # Git exclusions (models, datasets, venv, node_modules)
└── README.md             # Master project documentation
```

---

## 🗺 Roadmap

- [x] U-Net model architecture
- [x] Training pipeline with mixed precision & DataParallel
- [x] Hardened FastAPI inference API with DoS protection
- [x] Model hosting via GitHub Releases
- [x] React 19 + Three.js multispectral web frontend
- [x] Image return (RGB composite, mask overlay, probability heatmap)
- [ ] Batch prediction endpoint
- [ ] Docker containerization (`docker-compose`)
- [ ] Cloud deployment (GCP Cloud Run / AWS ECS)

---

## 🤝 Contributing

Contributions are welcome! Please check out the issues or open a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- **38-Cloud Dataset** by Sorour Mohajerani and Parvaneh Saeedi ([Paper](https://arxiv.org/abs/1901.03358))
- **U-Net Architecture** by Ronneberger et al. ([Paper](https://arxiv.org/abs/1505.04597))
- **USGS / NASA** for Landsat 8 multispectral imagery