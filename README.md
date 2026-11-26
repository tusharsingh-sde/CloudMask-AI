# CloudMask AI — Multispectral Satellite Cloud Segmentation Workbench

> **Real-time 4-band Landsat 8 cloud segmentation powered by a PyTorch U-Net, served through a hardened in-memory FastAPI engine and an interactive mission-control workbench.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Inference-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-In--Memory_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Vite_+_Three.js-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose_+_Nginx-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

**Live Demo:** [https://cloud-mask-ai.vercel.app](https://cloud-mask-ai.vercel.app) · **Container Registry:** `ghcr.io/tusharsingh-sde/cloudmask-backend`

<!-- Drop your 2 best screenshots into docs/screenshots/ and uncomment:
![CloudMask AI Workbench — Partial Cloud Split Compare](docs/screenshots/partial-cloud-split.png)
-->

---

## 1. The Problem: Why RGB Alone Fails in Earth Observation

Over **65% of Earth's surface** is obscured by clouds at any given moment. In remote sensing pipelines (agriculture monitoring, flood mapping, urban change detection), unmasked clouds corrupt downstream analytics and produce false spectral readings.

Standard computer vision models trained only on 3-channel RGB images fail on one critical edge case: **bright snow and ice look almost identical to clouds in visible light.**

**CloudMask AI** solves this by ingesting **4 synchronized 16-bit Landsat 8 OLI spectral bands**:

| Band | Channel | Wavelength | Role in Segmentation |
| :--- | :--- | :--- | :--- |
| **B04** | Red | `640 – 670 nm` | Visible cloud brightness & soil contrast |
| **B03** | Green | `530 – 590 nm` | Vegetation & surface reflectance |
| **B02** | Blue | `450 – 515 nm` | Atmospheric scattering & thin haze |
| **B05** | **Near-Infrared (NIR)** | `850 – 880 nm` | **Separates reflective snow/ice terrain from dense cloud cover** |

---

## 2. Verified Inference Benchmarks (38-Cloud Dataset)

Evaluated on real 16-bit `384 × 384` (`147,456 px`) Landsat 8 patches from the **38-Cloud** dataset using the production Docker CPU container (`Epoch 11` checkpoint, validation loss `0.0958`, confidence threshold `50%`):

| Curated Scene | Patch ID | Ground Truth | Model Prediction | Clear Sky / Ground | CPU Latency | What It Proves |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Partial Cloud** | `patch_113` | ~56% | **57.06%** | 42.94% | `1244.1 ms` | Sharp wispy cloud boundary segmentation over dark terrain |
| **Snow vs. Cloud** | `patch_291` | ~0% | **0.38%** | 99.62% | `1270.9 ms` | **NIR false-positive rejection** on snow/ice-covered land & rivers |
| **Heavy Overcast** | `patch_157` | ~90% | **94.24%** | 5.76% | `1230.9 ms` | Dense multi-layer cloud detection while preserving clear gaps |

> All 3 patches are bundled directly inside the frontend so anyone visiting the live app can reproduce these exact results in **1 click** without hunting for raw GeoTIFF files.

---

## 3. System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite + Three.js)                                  │
│  • 3D Earth Hero Overview & Sticky Live-Telemetry Navbar             │
│  • 1-Click "Live Demo" + Non-Blocking 3-Step Guided Coachbar         │
│  • Bundled Landsat 8 Sample Scenes + Multi-Band Drag & Drop          │
│  • Split-Curtain Compare · Single View · 2×2 Quad Grid · PNG Export  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ POST /api/predict (4 × 16-bit GeoTIFF)
                                  │ ?threshold=0.5&overlay_opacity=0.45
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Nginx Reverse Proxy (Local Docker / Vercel)                         │
└─────────────────────────────────┬────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI Inference Engine (Python 3.12 · Non-Root)                   │
│  1. Validate & decode 4 raw 16-bit bands strictly in memory          │
│  2. Radiometric normalization → 4-channel tensor [1, 4, H, W]       │
│  3. PyTorch U-Net forward pass (CPU wheel, weights_only=True)        │
│  4. Sigmoid thresholding + pixel telemetry calculation               │
│  5. Render 4 visual artifacts: RGB, Cyan Overlay, Binary Mask,       │
│     and Magma Probability Heatmap (zero temporary disk writes)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Features

### Interactive Ground-Station UX

- **Zero-Friction ▶ Live Demo + 3-Step Guided Coachbar:** Clicking *Live Demo* in the navbar scrolls smoothly to the workbench, auto-loads all 4 bands of the Partial Cloud scene, triggers live U-Net inference, and walks the visitor through curtain comparison and a 1-click Snow vs. Cloud NIR stress test.
- **1-Click Sample Scenes:** Built-in curated Landsat 8 scenes (Partial Cloud, Snow vs. Cloud, Heavy Overcast) with normalized RGB thumbnails for instant testing.
- **Smart Multi-Band Ingestion:** Drop all 4 `.tif` files at once—filenames (`red`/`b04`, `green`/`b03`, `blue`/`b02`, `nir`/`b05`) are auto-matched into their respective slots. Non-browser-renderable 16-bit GeoTIFFs render clean `16-BIT GEOTIFF · READY` status cards instead of broken thumbnails.
- **Multi-Mode Inspection Viewport:**
  - **Split Compare:** Interactive vertical curtain slider (pointer + ← / → keyboard accessible) comparing True-Color RGB against Mask Overlay, Binary Mask, or Magma Probability Heatmap.
  - **Single View & 2×2 Quad Grid:** Inspect all 4 output channels simultaneously, open fullscreen lightbox, switch between previous runs in Run History, or export individual / all 4 PNGs.
- **Live Inference Hyperparameters:** Tune *Confidence Threshold* (10% – 90%) for precision/recall control and *Overlay Opacity* (10% – 100%) for visual inspection.

### Production & ML Engineering

- **Zero Disk Latency:** Uploaded bands and output PNGs are processed completely in memory—no temporary files left on disk.
- **Safe & Lean PyTorch Runtime:** Uses CPU-only PyTorch wheels on `python:3.12-slim` under a non-root container user (`appuser`) and loads model checkpoints safely with `weights_only=True`.
- **Health & Readiness Probes:** Dedicated `/api/health` and `/api/ready` endpoints wired to Docker healthchecks and frontend live telemetry polling.
- **Automated Quality Gates:** Backed by a `pytest` backend test suite, `oxlint` + Vite frontend checks, and GitHub Actions CI.

---

## 5. Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning** | PyTorch (4-Channel U-Net), NumPy, 38-Cloud Landsat 8 Dataset |
| **Backend API** | FastAPI, Uvicorn, Pillow / GeoTIFF Processing, In-Memory Pipelines |
| **Frontend** | React, Vite, Three.js (EarthScene), Lucide Icons, Custom Mission-Control CSS |
| **Containerization & Proxy** | Docker, Docker Compose, Nginx (`nginx:alpine`), Multi-Stage Builds |
| **Testing & Deployment** | Pytest, GitHub Actions CI, GHCR, Azure Container Apps (Scale-to-Zero), Vercel |

---

## 6. Project Structure

```text
CloudMask-AI/
├── backend/
│   ├── Dockerfile              # Lean CPU-only python:3.12-slim non-root image
│   ├── requirements.txt
│   ├── download_model.py       # Optional checkpoint fetch for cloud builds
│   └── ...                     # FastAPI app, U-Net architecture, inference & tests
├── frontend/
│   ├── public/samples/         # 3 curated 4-band Landsat 8 sample scenes (~3.5 MB)
│   │   ├── partial-cloud/      # red.tif, green.tif, blue.tif, nir.tif, preview.png
│   │   ├── snow-vs-cloud/
│   │   └── heavy-cloud/
│   ├── src/
│   │   ├── components/         # Navbar, HeroSection, EarthScene, StudioWorkbench
│   │   ├── constants/          # bands.js, samples.js
│   │   ├── services/api.js     # /api/health & /api/predict client
│   │   └── App.jsx             # Shared demo state & 3-step guided coachbar
│   ├── nginx.conf              # Reverse proxy (/api -> backend:8000)
│   └── Dockerfile              # Multi-stage Node 20 -> Nginx Alpine
├── docker-compose.yml          # Full-stack local orchestration with healthchecks
└── .github/workflows/          # Automated CI pipeline
```

---

## 7. Quick Start

### Option A: Docker Compose (Recommended — Full Stack in 1 Command)

```bash
git clone https://github.com/tusharsingh-sde/CloudMask-AI.git
cd CloudMask-AI
docker compose up --build
```

- **Frontend (via Nginx):** http://localhost:8080
- **Backend Readiness Check:** http://localhost:8000/api/ready
- Click **▶ Live Demo** in the top bar to verify the full 4-band pipeline immediately.

### Option B: Local Development (Without Docker)

**Backend:**

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173 (ensure `VITE_API_URL` points to `http://localhost:8000` if not using the Vite proxy).

---

## 8. API Reference

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Liveness status, active compute device (CPU), checkpoint epoch & loss |
| `/api/ready` | `GET` | Readiness probe returning `200 OK` only once U-Net weights are loaded |
| `/api/predict` | `POST` | Runs 4-band cloud segmentation and returns telemetry + 4 rendered image URLs |

### `POST /api/predict` Parameters:

- **Form Data** (`multipart/form-data`): `red`, `green`, `blue`, `nir` (synchronized `.tif`/`.tiff`/`.png`/`.jpg` band files)
- **Query Params:** `threshold` (0.10–0.90, default 0.50), `overlay_opacity` (0.10–1.00, default 0.45)

**Sample JSON Response (truncated):**

```json
{
  "cloud_coverage_percentage": 57.06,
  "clear_sky_percentage": 42.94,
  "cloud_pixels": 84143,
  "total_pixels": 147456,
  "rgb_composite_url": "data:image/png;base64,...",
  "overlay_url": "data:image/png;base64,...",
  "mask_url": "data:image/png;base64,...",
  "heatmap_url": "data:image/png;base64,...",
  "metadata": {
    "dimensions": "384 x 384 px",
    "inference_time_ms": 1244.1,
    "device": "cpu",
    "threshold": 0.5,
    "overlay_opacity": 0.45,
    "model_epoch": 11,
    "model_best_loss": 0.0958
  }
}
```

---

## 9. Cloud Deployment Architecture

Designed for low-cost, zero-idle-spend production hosting:

- **Container Registry (GHCR):** Backend Docker image built with model weights baked in and published to `ghcr.io/tusharsingh-sde/cloudmask-backend:v1`.
- **Backend Compute (Azure Container Apps):** 1.0 vCPU, 2.0 GiB RAM, port 8000, readiness probe `/api/ready`, min-replicas: 0 (scale-to-zero), with `ALLOWED_ORIGINS` locked to the production frontend domain.
- **Frontend Hosting (Vercel):** Deployed from `frontend/` with `VITE_API_URL` pointing to the Azure Container Apps HTTPS endpoint.

---

## Author

Built by **Tushar Singh** · [GitHub](https://github.com/tusharsingh-sde)
