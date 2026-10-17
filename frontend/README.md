# CloudMask AI - Multispectral Frontend

Interactive React 19 web application for satellite multispectral band cloud detection, featuring a real-time 3D orbital Three.js Earth simulation, drag-and-drop band uploads, and deep learning segmentation analysis.

## 🚀 Features

- **3D Orbital Earth Simulation**: Interactive WebGL scene rendered with Three.js showing orbital trajectories and satellite fleet.
- **Multispectral Ingestion**: Dedicated dropzones for Red (B04), Green (B03), Blue (B02), and Near-Infrared (B08) bands.
- **Multi-View Output**:
  - Normalized True-Color RGB Composite
  - Predicted U-Net Binary Cloud Mask
  - Neon Magenta Overlay on True-Color Satellite Imagery
  - Continuous Magma Colormap Probability Heatmap
- **Interactive Controls**: Live sliders for Confidence Threshold (10%–90%) and Overlay Opacity.
- **Direct Exports**: One-click high-resolution PNG downloads for all generated visual products.
- **Reverse Proxy**: Pre-configured Vite proxy forwarding `/api` calls directly to FastAPI on `127.0.0.1:8000`.

## 🛠️ Setup & Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment (optional, defaults to reverse proxy):
   ```bash
   cp .env.example .env
   ```

3. Launch development server:
   ```bash
   npm run dev
   ```

4. Build for production:
   ```bash
   npm run build
   ```
