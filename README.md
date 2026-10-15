<div align="center">

# ☁️ CloudMask AI

**Automated Cloud Segmentation for Satellite Imagery using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Pixel-level cloud detection in Landsat 8 multispectral satellite imagery using a U-Net architecture trained on the 38-Cloud dataset.*

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Demo](#-demo) · [Contributing](#-contributing)

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
- [Demo](#-demo)
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
- A **REST API** for integrating cloud detection into any application
- A **Streamlit testing interface** for visual inspection and rapid prototyping
- A **web frontend** (coming soon) for production deployment

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **U-Net Segmentation** | Custom encoder-decoder architecture with skip connections for precise pixel-level cloud masks |
| ⚡ **Mixed Precision Training** | FP16 training with `torch.amp` for 2x faster training and lower VRAM usage |
| 🔥 **Multi-GPU Support** | Automatic `DataParallel` scaling across multiple GPUs |
| 🌐 **REST API** | FastAPI backend with async image processing |
| 🖥️ **Streamlit UI** | Interactive testing dashboard with mask overlay visualization |
| 📊 **Cloud Coverage Stats** | Quantitative cloud coverage percentage per image |
| 🌡️ **Probability Heatmaps** | Confidence visualization beyond binary masks |
| 💾 **Smart Checkpointing** | Per-epoch, latest, and best model saves with full resume support |
| 🔄 **Cross-Platform** | Runs on Windows, Linux, and Kaggle environments |

---

## 🏗 Architecture

### Model — U-Net

```
Input (4, 384, 384)          Output (1, 384, 384)
   │                              ▲
   ▼                              │
┌──────────┐                ┌──────────┐
│ Encoder  │───Skip Conn───▸│ Decoder  │
│ 64→128   │                │ 128→64   │
│ →256→512 │                │ 256→128  │
└────┬─────┘                └────┬─────┘
     │                           ▲
     ▼                           │
  ┌──────────────────────────────┐
  │     Bottleneck (1024)        │
  └──────────────────────────────┘
```

- **Input**: 4-channel image (Red, Green, Blue, NIR) — 384×384 patches
- **Encoder**: 4 downsampling blocks with `DoubleConv` (Conv → BN → ReLU × 2) + MaxPool
- **Bottleneck**: 1024-channel feature representation
- **Decoder**: 4 upsampling blocks with `ConvTranspose2d` + skip connections
- **Output**: Single-channel binary cloud mask (logits)

### System Architecture

```
┌────────────────────────────────────────────────────────┐
│                    CloudMask AI                        │
├──────────────┬──────────────┬──────────────────────────┤
│  Frontend    │   Backend    │     ML Pipeline          │
│  (Coming     │   FastAPI    │                          │
│   Soon)      │   REST API   │  Dataset → Train → Model │
│              │              │                          │
│  React /     │  /predict    │  38-Cloud   UNet   .pt   │
│  Next.js     │  /health     │  Dataset    FP16   Ckpt  │
├──────────────┤              ├──────────────────────────┤
│  Streamlit   │              │                          │
│  Test UI     │◄────────────►│  Inference Engine        │
└──────────────┴──────────────┴──────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch, torchvision |
| **Computer Vision** | OpenCV, PIL |
| **Backend API** | FastAPI, Uvicorn |
| **Testing UI** | Streamlit |
| **Frontend** | React / Next.js *(coming soon)* |
| **Training Infra** | CUDA, AMP (Mixed Precision), DataParallel |
| **Dataset** | [38-Cloud (Landsat 8)](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU *(recommended for training, CPU works for inference)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/tusharsingh-sde/CloudMask-AI.git
cd CloudMask-AI

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the pre-trained model
python download_model.py
```

### Quick Start

**Run the Streamlit UI:**
```bash
streamlit run app.py
```

**Run the API server:**
```bash
python api.py
# Server starts at http://localhost:8000
```

---

## 🏋️ Training

### Dataset Setup

Download the [38-Cloud dataset](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) and place it in the project root:

```
CloudMask-AI/
└── 38-Cloud_training/
    ├── train_red/
    ├── train_green/
    ├── train_blue/
    ├── train_nir/
    └── train_gt/
```

> **Kaggle Users**: The dataset path is auto-detected. No manual setup needed.

### Run Training

```bash
python train.py
```

**Training Configuration:**

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | With weight decay 1e-4 |
| Learning Rate | 3e-4 | Tuned for batch size 32 |
| Batch Size | 32 (GPU) / 4 (CPU) | Auto-adjusted per device |
| Epochs | 20 | |
| Loss Function | BCEWithLogitsLoss | Numerically stable binary cross-entropy |
| Precision | FP16 (Mixed) | Via `torch.amp.GradScaler` |

**Checkpoints saved after every epoch:**

| File | Description |
|---|---|
| `cloudmask_unet_epoch_N.pt` | Per-epoch checkpoint (never overwritten) |
| `cloudmask_unet_latest.pt` | Most recent epoch (for quick resume) |
| `cloudmask_unet_best.pt` | Lowest loss model (for deployment) |

---

## 📡 API Reference

### Health Check

```http
GET /
```

**Response:**
```json
{
  "status": "CloudMask AI Backend is running!"
}
```

### Predict Cloud Mask

```http
POST /predict
Content-Type: multipart/form-data
```

| Parameter | Type | Description |
|---|---|---|
| `red` | `file` | Red band image (.tif / .jpg / .png) |
| `green` | `file` | Green band image |
| `blue` | `file` | Blue band image |
| `nir` | `file` | Near-infrared band image |

**Response:**
```json
{
  "status": "success",
  "cloud_coverage_percentage": 42.37,
  "message": "Mask generated successfully"
}
```

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/predict \
  -F "red=@red_patch_001.TIF" \
  -F "green=@green_patch_001.TIF" \
  -F "blue=@blue_patch_001.TIF" \
  -F "nir=@nir_patch_001.TIF"
```

---

## 🖥 Demo

### Streamlit Testing Interface

The built-in Streamlit app provides a visual testing dashboard:

```bash
streamlit run app.py
```

**Features:**
- Upload 4 satellite band images (Red, Green, Blue, NIR)
- Adjustable cloud confidence threshold
- Visual outputs: RGB composite, predicted mask, mask overlay
- Cloud probability heatmap
- Real-time cloud coverage statistics

---

## 📁 Project Structure

```
CloudMask-AI/
│
├── model.py              # U-Net architecture definition
├── dataset.py            # 38-Cloud dataset loader (4-band + ground truth)
├── train.py              # Training loop (AMP, multi-GPU, checkpointing)
│
├── api.py                # FastAPI REST API for inference
├── app.py                # Streamlit testing UI
│
├── download_model.py     # Auto-download model from GitHub Releases
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusions (models, data, venv)
└── README.md             # You are here
```

---

## 🗺 Roadmap

- [x] U-Net model architecture
- [x] Training pipeline with mixed precision
- [x] Multi-GPU support (DataParallel)
- [x] FastAPI inference API
- [x] Streamlit testing dashboard
- [x] Model hosting via GitHub Releases
- [ ] React / Next.js frontend
- [ ] Mask image return in API response
- [ ] Batch prediction endpoint
- [ ] Model performance metrics (IoU, F1, Precision, Recall)
- [ ] Docker containerization
- [ ] Cloud deployment (GCP / AWS)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **38-Cloud Dataset** — [Sorour Mohajerani et al.](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) for the Landsat 8 cloud segmentation dataset
- **U-Net** — [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597) for the original U-Net architecture

---

<div align="center">

**Built with ❤️ by [Tushar Singh](https://github.com/tusharsingh-sde)**

</div>