# CloudMask AI — Frontend Redesign Specification & Implementation Plan
**Branch:** `feat/immersive-earth-frontend`  
**Target:** React 19 + Vite 8 + Three.js 0.186  
**Audience:** Claude Opus / Senior Frontend Engineer  
**Status:** Implementation-Ready Specification (Preparation Complete — Implementation Not Started)

---

## 1. Verified Repository Audit & Technical Baseline

This section distinguishes between **verified facts** inspected directly from active source code and **unverified claims or assumptions**, followed by critical engineering decisions.

### 1.1 Verified Repository Facts
- **Git State:** 
  - Repository root is `c:\Users\tstus\Desktop\CloudMask-AI`.
  - Remote tracking: `https://github.com/tusharsingh-sde/CloudMask-AI.git`.
  - Active branch is `feat/immersive-earth-frontend` (branched from `main` commit `4de52a5`).
  - Working tree is clean except for the untracked `docs/` directory.
- **Frontend Stack & Dependencies (`frontend/package.json`):**
  - `react`: `^19.2.8`
  - `react-dom`: `^19.2.8`
  - `three`: `^0.186.1` (native Three.js, not React Three Fiber)
  - `lucide-react`: `^1.59.0`
  - `@vitejs/plugin-react`: `^6.1.1`
  - `vite`: `^8.3.0`
  - `oxlint`: `^1.81.0`
  - Node packages are installed in `frontend/node_modules`.
- **Linting & Compilation Status:**
  - `npm run lint` (`oxlint`): **0 errors, 0 warnings** across all 10 frontend files.
  - `npm run build` (`vite build`): Compiles successfully in ~1.9s.
  - No automated unit test runner (Vitest/Jest) or E2E runner (Playwright) is currently configured.
- **Vite Configuration (`frontend/vite.config.js`):**
  - Dev server runs on port `5173`.
  - Reverse proxy: `/api` maps to `http://127.0.0.1:8000` (`changeOrigin: true`, `secure: false`).
- **Navigation & Component Structure:**
  - `frontend/src/main.jsx` mounts `App` to `#root`.
  - `frontend/src/App.jsx` handles state-based view switching via `useState("landing")` between three views: `"landing"`, `"processing"`, `"result"`. No router library is used.
  - Page state, selected files, analysis results, and threshold/opacity settings live at the root of `App.jsx`.
- **Backend API Implementation (`backend/api.py`):**
  - FastAPI service on port `8000`.
  - `GET /api/health` returns:
    ```json
    {
      "status": "online",
      "service": "CloudMask AI Backend",
      "device": "cuda" | "cpu",
      "model_loaded": boolean,
      "checkpoint": { "epoch": int | "unknown", "best_loss": float }
    }
    ```
  - `POST /api/predict` (and aliases `/predict`, `/api/process`) expects `multipart/form-data`:
    - `red`: File (required)
    - `green`: File (required)
    - `blue`: File (required)
    - `nir`: File (required)
    - `threshold`: Optional float Form parameter (default: `0.5`, clamped in backend to `[0.05, 0.95]`)
    - `overlay_opacity`: Optional float Form parameter (default: `0.45`, clamped in backend to `[0.1, 1.0]`)
  - Enforced file constraints in backend:
    - Max size: 25 MB per band (`MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024`).
    - Allowed extensions: `.tif`, `.tiff`, `.jpg`, `.jpeg`, `.png`.
    - Allowed dimensions: 16x16 px to 4096x4096 px.
    - Decoded in-memory via OpenCV/tifffile, channels harmonized to red band's dimensions, normalized to uint8.
  - Generated output payload in backend:
    - `status`: `"success"`
    - `cloud_coverage_percentage`: float
    - `clear_sky_percentage`: float
    - `cloud_pixels`: int
    - `clear_pixels`: int
    - `total_pixels`: int
    - `rgb_composite_url`: Base64 PNG data URL
    - `mask_url`: Base64 PNG data URL
    - `overlay_url`: Base64 PNG data URL (neon magenta `[255, 30, 140]` over RGB)
    - `heatmap_url`: Base64 PNG data URL (OpenCV Magma colormap)
    - `imageUrl`: Base64 PNG data URL (alias pointing to `overlay_url`)
    - `metadata`: `{ dimensions, threshold, overlay_opacity, device, inference_time_ms, bands_analyzed, model_epoch, model_best_loss }`
    - `message`: string
- **Asset Reality:**
  - `frontend/public/` contains only `favicon.svg` and `icons.svg`.
  - `frontend/src/assets/` is completely empty.
  - Existing `EarthScene.jsx` loads 4 texture maps over external HTTP from Three.js examples:
    - `https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg`
    - `https://threejs.org/examples/textures/planets/earth_normal_2048.jpg`
    - `https://threejs.org/examples/textures/planets/earth_specular_2048.jpg`
    - `https://threejs.org/examples/textures/planets/earth_clouds_1024.png`
  - **There are no local Earth texture files in the repository.**

### 1.2 Claims Remaining Unverified
- **Offline Reliability of External Textures:** External textures rely on public network availability and Three.js CDN uptime. If the user runs the app offline, texture loading will fail unless a procedural fallback or local asset cache is available.
- **Browser WebGL Performance Across Hardware:** Three.js scene runs with `powerPreference: "high-performance"` and `devicePixelRatio` capped to 2; performance on lower-tier integrated GPUs has not been benchmarked.
- **True TIF/GeoTIFF Browser Display:** Browser engines cannot display raw 16-bit GeoTIFF files natively in standard `<img>` tags before submission without specialized in-browser decoders. Selected TIFF bands prior to backend processing cannot be displayed as live image previews; they must be clearly represented via stylized technical file-status cards (filename, format badge, file size in KB).

### 1.3 Key Architectural Decisions
1. **No React Three Fiber (R3F):** Remain on native Three.js `0.186.1` encapsulated inside React hooks (`useRef`, `useEffect`). Avoids peer dependency conflicts with React 19 and maintains tight memory cleanup control.
2. **Robust Earth Asset & Fallback Strategy:** Asynchronously load high-resolution NASA Blue Marble / Three.js planet textures with clear progress handling, backed by an immediate, elegant stylized celestial canvas fallback (deep navy/oceanic gradient with soft atmospheric glow). The globe will never render as a broken wireframe, black sphere, or missing texture.
3. **Preserve Internal State Routing:** Keep `useState` view switching in `App.jsx` rather than introducing `react-router-dom`. It is completely reliable, lightweight, and prevents routing state bugs.

---

## 2. Three-Page Product Scope & UX Specification

The application consists of **exactly three distinct pages** sharing a cohesive deep-space design system.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Page 1: Cinematic Landing Page                                         │
│ • Minimal branding, strong headline, short copy                        │
│ • Exactly ONE primary CTA ("Start Analysis")                           │
│ • Realistic 3D Earth on right with interactive drag-to-rotate          │
│ • 4 realistic satellites in Keplerian orbits                           │
│ • Deep space navy/black with subtle celestial particle mesh            │
│ • STRICTLY NO fake telemetry, status pills, tickers, or metric badges │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ User clicks "Start Analysis"
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Page 2: Multispectral Upload Workspace                                 │
│ • 4 Band cards: Red (B04), Green (B03), Blue (B02), NIR (B08)          │
│ • Drag-and-drop, file picker, file size, replace & remove actions      │
│ • Compact threshold (10-90%) & overlay opacity (10-100%) controls      │
│ • "Run Cloud Segmentation" action (disabled until all 4 bands valid)   │
│ • Backend connection status displayed in workspace header/sidebar only │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Inference complete
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Page 3: Scientific Results Dashboard                                   │
│ • Real backend metrics: Cloud %, Clear %, Pixel counts, Latency        │
│ • 4 Layer views: Overlay, Binary Mask, RGB Composite, Probability Heat │
│ • Interactive before/after split slider comparison                     │
│ • Fullscreen inspection modal & PNG export for all available layers    │
│ • Discreet telemetry drawer: device, dimensions, model checkpoint epoch│
└────────────────────────────────────────────────────────────────────────┘
```

---

### Page 1 — Cinematic Landing Page

A minimalist, cinematic, deep-space Earth-observation experience that focuses entirely on visual impact and immediate accessibility.

#### Layout & Content
- **Asymmetric Split Layout:**
  - **Left Column (45% width on desktop):**
    - **Project Wordmark:** Clean, understated geometric glyph + `CloudMask AI` with modern typography.
    - **Dominant Headline:** *"Precision Cloud Segmentation for Satellite Imagery"* or *"See Earth in a Different Spectrum"*.
    - **Concise Supporting Text:** Exactly 2–3 short sentences explaining multispectral satellite cloud detection across Red, Green, Blue, and NIR bands powered by a trained deep neural network.
    - **Exactly One Primary CTA:** A single high-visibility button labeled `Start Analysis` with a subtle arrow icon and restrained cyan accent glow. Clicking transitions directly to Page 2 (Upload Workspace).
  - **Right Column (55% width on desktop):**
    - Large, photorealistic 3D Earth positioned prominently on the right.
    - Four animated satellites orbiting on distinct orbital planes.
  - **Background:**
    - Deep space void (`#02040a` to `#060b19`).
    - Sparse celestial star points and a subtle, restrained cyan particle mesh concentrated predominantly behind the Earth to provide spatial depth without visual noise.

#### Earth & Orbital Interaction
- **Realistic Globe:** Continents, oceans, specular water sheen, animated cloud layer, and a restrained cyan atmospheric glow.
- **Tilted Axis & Continuous Spin:** Slow auto-rotation around Earth's natural axial tilt (~23.4°).
- **Interactive Drag-to-Rotate:**
  - Pointer down + drag allows horizontal (yaw) and vertical (pitch) rotation of the Earth.
  - Pitch rotation clamped to `[-Math.PI / 4, Math.PI / 4]` to prevent disorienting polar flips.
  - On pointer release, momentum decays smoothly and automatic rotation gracefully resumes after a brief pause (~1.5s).
- **Satellite Constellation:**
  - 4 visually credible satellites constructed from lightweight standard Three.js primitives (avionics cylinder, gold/navy solar arrays, communications dish).
  - Satellites travel along 4 distinct elliptical orbital rings with delicate cyan luminescence (`rgba(0, 242, 254, 0.25)`).

#### Strictly Prohibited on Page 1
- **NO backend status pill or health badge.** (Status monitoring belongs exclusively in the workspace/upload header and sidebar).
- **NO live statistics or invented telemetry.**
- **NO scrolling ticker ribbons or feature badge lists.**
- **NO decorative metric cards, multiple CTAs, or HUD widgets.**
- **NO loud neon magenta gradients or distracting particle vortexes.**

---

### Page 2 — Multispectral Upload Workspace

A clean, functional, and uncluttered scientific data entry environment for four satellite spectral bands.

#### Band Configuration (Landsat 8 / Sentinel-2 Standards)
1. **Red Band — B04** (640–670 nm): Visible red channel.
2. **Green Band — B03** (530–590 nm): Visible green channel.
3. **Blue Band — B02** (450–515 nm): Visible blue channel.
4. **Near-Infrared (NIR) Band — B08** (850–880 nm): Key band for distinguishing bright clouds from high-albedo ground and snow.

#### UX & Functional Requirements
- **Header:** Workspace breadcrumbs, clean title, and backend connection status indicator (`BACKEND ONLINE` / `CONNECTING`).
- **4 Upload Cards / Dropzones:**
  - Clean card grid with spectral badge (`B04`, `B03`, `B02`, `B08`) and band title.
  - Drag-and-drop zone and standard file browser trigger.
  - Accepted extensions: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`.
  - Max file size: 25 MB per band (strictly matching backend limit).
- **File State Management Per Card:**
  - **Empty State:** Clean dashed border, upload icon, drag instruction, and allowed file formats hint.
  - **Selected / Valid State:** Card border highlights with band-specific accent; displays original filename, formatted file size (KB/MB), format pill, and a ready indicator.
  - **Individual Replace & Remove Actions:**
    - **Replace Button:** Re-triggers the file picker for that specific card.
    - **Remove Button:** Clears only that band's file without resetting other uploaded bands.
  - **Invalid File State:** Immediate feedback if file extension is unsupported or file size exceeds 25 MB.
- **Compact Inference Parameter Controls:**
  - **Confidence Threshold:** Compact slider from `10%` to `90%` (step `5%`, default `50%`), controlling the classification decision boundary.
  - **Overlay Opacity:** Compact slider from `10%` to `100%` (step `5%`, default `45%`), controlling the visibility of the cloud mask over the RGB composite.
  - Sliders styled cleanly with cyan tracking and monospace value readouts.
- **Primary Submission Action:**
  - Prominent button labeled `Run Cloud Segmentation`.
  - Strictly disabled until all 4 bands have valid files selected.
  - Displays animated spinner and status *"Analyzing Satellite Bands..."* during inference.
  - Clear error banner if the network request fails or backend returns an HTTP error.

---

### Page 3 — Scientific Results Dashboard

A clean analytical workspace prioritizing genuine imagery, side-by-side comparison, and real deep learning inference metrics.

#### 1. Quantitative KPI Banner
Displays real numbers directly calculated by the PyTorch model:
- **Cloud Coverage:** Exact percentage (e.g. `24.38%`) with a subtle proportional bar.
- **Clear Sky / Surface:** Exact percentage (e.g. `75.62%`).
- **Pixel Breakdown:** Real counts for Cloud Pixels, Clear Pixels, and Total Processed Pixels.
- **Inference Latency:** Elapsed time reported by backend (e.g. `42.6 ms`).
- **Compute Device:** Device utilized (`CUDA` or `CPU`).

#### 2. Imagery Display & Layer Switching
Four distinct visualization modes directly backed by backend base64 PNG responses:
1. **Mask Overlay (`overlay_url`):** High-visibility cloud mask overlaid on the normalized true-color ground imagery.
2. **Binary Cloud Mask (`mask_url`):** Raw neural network output (white = detected cloud, black = clear surface).
3. **RGB Composite (`rgb_composite_url`):** True-color composite rendered from the uploaded Red, Green, and Blue bands.
4. **Probability Heatmap (`heatmap_url`):** Continuous inference probability visualized using the OpenCV Magma colormap.

#### 3. Interactive Comparison & Inspection
- **Interactive Split-Slider:** Allows horizontal dragging across the viewport to compare the raw True-Color RGB composite directly against the Mask Overlay.
- **Fullscreen Lightbox:** Click-to-expand modal for close inspection of high-resolution satellite imagery.
- **Layer Export:** Individual PNG download buttons for each visualization layer (`cloudmask_overlay.png`, `cloudmask_binary_mask.png`, etc.) using real blob downloads.
- **Telemetry Drawer:** Discreet panel detailing tensor resolution, threshold applied, model checkpoint epoch, and validation loss.
- **Reset Action:** `Analyze New Scene` button returning cleanly to the Upload Workspace while maintaining backend connection.

#### Strict Truth-in-Data Constraints
- **Never fabricate metrics or metadata.** If the backend response lacks a field, display `N/A` or omit it.
- **Never claim to reconstruct a cloud-free image.** The pipeline performs cloud detection and segmentation, not generative inpainting or cloud removal.

---

## 3. Shared Design System Specification

### 3.1 Color Palette & Tokens
```css
:root {
  /* Surface & Backgrounds */
  --color-space-abyss: #02040a;      /* Page background */
  --color-space-navy: #060b19;       /* Deep secondary background */
  --color-surface-panel: rgba(8, 15, 30, 0.75); /* Frosted card surface */
  --color-surface-hover: rgba(14, 25, 48, 0.85);

  /* Hairline Borders */
  --border-subtle: rgba(56, 189, 248, 0.12);
  --border-active: rgba(56, 189, 248, 0.35);
  --border-hover: rgba(56, 189, 248, 0.25);

  /* Accents & Luminescence */
  --color-cyan-electric: #00f2fe;   /* Primary interaction accent */
  --color-cyan-glow: #06b6d4;       /* Secondary cyan */
  --color-ice-white: #f8fafc;        /* High-contrast text */
  --color-text-secondary: #94a3b8;   /* Muted captions & labels */
  --color-text-muted: #64748b;       /* Metadata keys */

  /* Spectral Band Identifiers */
  --band-red: #f87171;
  --band-green: #4ade80;
  --band-blue: #38bdf8;
  --band-nir: #c084fc;

  /* Status Colors */
  --status-online: #10b981;
  --status-offline: #f59e0b;
  --status-error: #ef4444;

  /* Shadows & Glow */
  --glow-cyan-sm: 0 0 12px rgba(0, 242, 254, 0.15);
  --glow-cyan-md: 0 0 24px rgba(0, 242, 254, 0.22);
}
```

### 3.2 Typography & Aesthetics
- **Headings & Body Copy:** Clean, modern sans-serif: `Inter`, system `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto`. Tight tracking on titles (`letter-spacing: -0.02em`).
- **Data & Telemetry:** Monospace font for coordinates, file sizes, percentages, and latencies: `"JetBrains Mono", "DM Mono", "Consolas", monospace`.
- **Restraint:** 1px hairline borders, subtle glassmorphic backdrop filters (`backdrop-filter: blur(16px)`), and smooth 200ms transitions. No garish saturated glows.

---

## 4. Proposed Frontend Component & File Architecture

To guarantee zero regression, new modular components will be placed in cleanly separated directories under `src/components/`, allowing the existing working files to remain intact as safe fallbacks throughout development.

```
frontend/src/
├── components/
│   ├── common/
│   │   ├── Header.jsx             # Top header for workspace/results with wordmark & status indicator
│   │   └── StatusIndicator.jsx    # Real-time backend connection status dot & label (workspace/results only)
│   ├── 3d/
│   │   ├── EarthCanvas.jsx        # Three.js viewport container with resize & pointer drag handling
│   │   ├── EarthGlobe.js          # Three.js Earth sphere, materials, texture loader & stylized fallback
│   │   ├── Satellites.js          # 4 procedural satellite meshes & orbital trajectories
│   │   └── ParticleMesh.js        # Subtle celestial cyan particle mesh
│   ├── landing/
│   │   └── LandingView.jsx        # Minimal cinematic hero: wordmark, headline, copy, "Start Analysis" CTA
│   ├── upload/
│   │   ├── UploadWorkspace.jsx    # Complete 4-band upload page view
│   │   ├── BandCard.jsx           # Individual band dropzone with replace/remove actions
│   │   └── ParameterControls.jsx  # Compact threshold & opacity sliders
│   ├── results/
│   │   ├── ResultsDashboard.jsx   # Results page view orchestrator
│   │   ├── MetricsSummary.jsx     # Cloud %, clear %, pixel counts, latency
│   │   ├── ImageStage.jsx         # Layer tabs, split comparison slider & fullscreen modal
│   │   └── TelemetryCard.jsx      # Discreet model & device telemetry breakdown
│   │
│   │   # Preserved existing files (untouched fallback baseline):
│   ├── LandingPage.jsx            # (Existing)
│   ├── ProcessingDashboard.jsx    # (Existing)
│   ├── ResultViewer.jsx           # (Existing)
│   ├── EarthScene.jsx             # (Existing)
│   └── Sidebar.jsx                # (Existing)
├── styles/
│   ├── tokens.css                 # Color tokens, typography, and glow variables
│   ├── space.css                  # Deep-space backgrounds and particle mesh styling
│   └── ui.css                     # Reusable cards, buttons, sliders, and modal styling
├── services/
│   └── api.js                     # UNTOUCHED existing API client
├── constants/
│   └── bands.js                   # UNTOUCHED spectral band constants
├── App.jsx                        # View switching between LandingView, UploadWorkspace, ResultsDashboard
├── index.css                      # Base reset & typography imports
└── main.jsx                       # Root mount
```

### Component Status Mapping
| File | Action | Rationale |
|---|---|---|
| `src/services/api.js` | **Leave Untouched** | Fully verified, correct timeout and payload contracts. |
| `src/constants/bands.js` | **Leave Untouched** | Correct spectral band metadata (B04, B03, B02, B08). |
| `src/components/common/*` | **Create New** | Clean, minimalist shared UI primitives. |
| `src/components/3d/*` | **Create New** | Decoupled Earth globe, satellite swarm, and drag interaction. |
| `src/components/landing/LandingView.jsx` | **Create New** | Clean cinematic landing strictly adhering to user guidelines (no status pill). |
| `src/components/upload/*` | **Create New** | Upload workspace with individual replace/remove actions. |
| `src/components/results/*` | **Create New** | Results dashboard with split-slider and real backend KPIs. |
| `src/App.jsx` | **Switch Views** | Direct view state to new components when validated. |
| `src/components/LandingPage.jsx` (etc.) | **Preserve** | Kept alongside new files during staging to avoid breaking working state. |

---

## 5. Earth Asset Strategy & 3D Implementation Mechanics

### 5.1 Earth Asset Sourcing, Licensing & Loading Strategy
- **Preferred Primary Asset Source:**
  - High-resolution textures from the Three.js examples planet collection:
    - Diffuse map: `https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg`
    - Normal bump map: `https://threejs.org/examples/textures/planets/earth_normal_2048.jpg`
    - Specular map: `https://threejs.org/examples/textures/planets/earth_specular_2048.jpg`
    - Clouds map: `https://threejs.org/examples/textures/planets/earth_clouds_1024.png`
- **Licensing & Attribution:**
  - These textures derive directly from the NASA Visible Earth / Blue Marble project, which is in the public domain and free for commercial and non-commercial scientific use. Three.js example code and packaging are MIT licensed.
  - Attribution note: *"Earth imagery courtesy of NASA Visible Earth / Goddard Space Flight Center"*.
- **Loading Behavior & Non-Blocking Mount:**
  - Load textures asynchronously using `THREE.TextureLoader` with callbacks (`onLoad`, `onProgress`, `onError`).
  - The 3D scene mounts immediately with standard base materials, so the user never encounters a blank screen or UI freeze while textures stream over HTTP.
  - As textures finish loading, they are mapped smoothly onto the existing Earth mesh with `needsUpdate = true`.
- **Realistic Fallback Quality (Grounded & Honest):**
  - **No false claims of photorealism for procedural fallbacks.** A procedural canvas cannot realistically duplicate complex continental coastlines, topography, and real atmospheric weather patterns.
  - **Instead, provide an elegant stylized celestial fallback:** If external textures fail to load (e.g. offline environment or CDN block), `EarthGlobe.js` applies an intentional, beautiful stylized deep-navy sphere (`#0c2340`) with a specular water sheen, faint latitude/longitude coordinate grid lines, and a soft cyan atmospheric glow (`#00f2fe`). The Earth renders as a polished, intentional scientific globe rather than a broken, untextured, or missing asset.

### 5.2 Implementation Proportionality (Simplicity Over Excessive Complexity)
- **Standard Three.js Shading:** Use standard `MeshPhongMaterial` with specular and normal maps. Avoid writing custom GLSL Rayleigh/Mie scattering vertex/fragment shaders; a concentric outer sphere with `THREE.BackSide`, additive blending, and gentle opacity achieves a stunning atmospheric halo with zero shader maintenance burden.
- **Parametric Keplerian Orbits (Simple & Robust):** Avoid complex N-body numerical integration or orbital mechanics libraries. Compute satellite positions using clean parametric trigonometric motion:
  ```javascript
  const x = Math.cos(angle) * radiusX;
  const y = Math.sin(angle) * radiusY;
  const z = Math.sin(angle * frequency + phase) * inclination;
  ```
  This guarantees rock-solid 60 FPS performance, zero mathematical divergence, and predictable visual paths across all devices.

### 5.3 Interactive Drag-to-Rotate Mechanics
- **Pointer Handling:**
  - `onPointerDown`: Capture pointer ID, store start `(clientX, clientY)`, set `isDragging = true`, pause auto-rotation.
  - `onPointerMove`: If dragging, compute `deltaX` and `deltaY`. Apply yaw rotation to `earthGroup.rotation.y` and pitch to `earthGroup.rotation.x`.
  - Pitch clamping: Clamp pitch strictly to `[-Math.PI / 4, Math.PI / 4]` to prevent inverted poles.
  - `onPointerUp` / `onPointerCancel`: Set `isDragging = false`.
  - Smooth Resume: Apply rotational inertia damping (`velocity * 0.92`). When velocity drops below `0.0001`, smoothly blend back to constant orbital rotation (`+0.0015 rad/frame`) after a 1.5-second idle delay.

### 5.4 Subtle Celestial Particle Mesh
- Three.js `Points` geometry composed of ~1,000 starlight vertices in a deep spherical shell (radius 18–35 units).
- Tinted pale ice-cyan (`#38bdf8`) and starlight white (`#ffffff`).
- Positioned predominantly behind and around the Earth to give depth without distracting from foreground readability.

---

## 6. Responsive & Performance Requirements

### Viewport Breakpoints
- **Desktop (>= 1200px):** Full 2-column split on landing; 4-column upload grid; top-metrics + split comparison stage on results.
- **Tablet / Small Laptop (768px – 1199px):**
  - Landing: Earth scales to 40% width or shifts slightly down.
  - Upload: 2x2 grid for band cards.
  - Results: Stacked split-slider with horizontal scroll for layer thumbnails.
- **Mobile (< 768px):**
  - Landing: Left column fills width; Earth rendered in upper background with lowered opacity (0.5) so text remains 100% legible. Single touch drag supported.
  - Upload: Single-column vertical stack with large touch dropzones.
  - Results: Full-width responsive image view with tabbed layer switcher.

### WebGL Memory & Performance Rules
1. **Strict Resource Disposal:** In `EarthCanvas.jsx` cleanup (`useEffect` return), systematically dispose of:
   - `renderer.dispose()`
   - All geometries (`geometry.dispose()`)
   - All materials (`material.dispose()`)
   - All textures (`texture.dispose()`)
2. **Device Pixel Ratio Cap:** `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))` to prevent performance dips on ultra-high-DPI retina screens.
3. **Animation Loop Cancellation:** Stop `renderer.setAnimationLoop(null)` on unmount.

---

## 7. Staged Implementation Instructions for Claude Opus

When executing the frontend redesign in a later phase, follow these five staged steps in strict order. **Work incrementally, test at every stage, and do not skip validation gates.**

### Mandatory Rule: No Unauthorized Actions
- **DO NOT install packages or change package versions without explicit approval.**
- **DO NOT modify backend or ML code.**
- **DO NOT execute `git commit` or `git push` without explicit user instruction.**

```
Stage 0: Time-Sensitive Pre-Flight Verification
   │ (Confirm branch, clean working tree, dependency status, API availability)
   ▼
Stage 1: Design Tokens & Cinematic Landing Page
   │ (Create styles/tokens.css, 3d/ components, landing/LandingView.jsx)
   ▼
Stage 2: Multispectral Upload Workspace
   │ (Create upload/ components with individual replace/remove & parameters)
   ▼
Stage 3: Scientific Results Dashboard
   │ (Create results/ components with real metrics, split-slider & export)
   ▼
Stage 4: App Integration, Verification & Polish
   │ (Wire App.jsx, run oxlint, test vite build, verify zero regressions)
```

### Stage 0: Time-Sensitive Pre-Flight Verification
- **Target Files:** Read-only inspection of repository state.
- **Tasks:**
  1. Verify active branch using `git branch --show-current`. If not on `feat/immersive-earth-frontend`, **STOP and report to user**.
  2. Inspect working tree using `git status`. If unexpected uncommitted changes exist, **STOP and report to user**. Do not switch branches or discard changes automatically.
  3. Verify dependencies in `frontend/package.json` are unchanged.
  4. Run `npm run lint` and `npm run build` in `frontend/` to re-verify the baseline.
- **Acceptance Gate:** 0 lint errors, clean build before touching any files.

### Stage 1: Design Tokens & Cinematic Landing Page
- **Target Files to Create:**
  - `frontend/src/styles/tokens.css` (color tokens, typography, hairline borders).
  - `frontend/src/components/3d/EarthGlobe.js` (Three.js Earth, materials, non-blocking texture loading, stylized fallback).
  - `frontend/src/components/3d/Satellites.js` (4 procedural satellites and elliptical orbits).
  - `frontend/src/components/3d/ParticleMesh.js` (subtle starlight & cyan particle shell).
  - `frontend/src/components/3d/EarthCanvas.jsx` (React wrapper with pointer drag-to-rotate handling & teardown).
  - `frontend/src/components/landing/LandingView.jsx` (minimal hero copy, exactly one `Start Analysis` CTA, zero status badges/tickers).
- **Dependencies:** Stage 0 complete.
- **Acceptance Gate:**
  - Landing renders cleanly with deep navy background, left-aligned text, and no status pills or decorative badges.
  - Earth rotates continuously; clicking and dragging rotates the globe smoothly in yaw and pitch.
  - Releasing pointer resumes auto-rotation after 1.5s idle.
  - 4 satellites orbit cleanly on separate orbital planes.
  - Clicking `Start Analysis` triggers navigation to the upload page.

### Stage 2: Multispectral Upload Workspace
- **Target Files to Create:**
  - `frontend/src/components/upload/BandCard.jsx` (individual dropzone for B04, B03, B02, B08 with replace and remove actions).
  - `frontend/src/components/upload/ParameterControls.jsx` (compact threshold slider 10-90%, opacity slider 10-100%).
  - `frontend/src/components/upload/UploadWorkspace.jsx` (complete upload view orchestrator with header status indicator).
- **Dependencies:** Stage 1 complete.
- **Acceptance Gate:**
  - 4 distinct band cards display correct labels and badges (Red B04, Green B03, Blue B02, NIR B08).
  - Drag-and-drop and file selection work for each band independently.
  - Replace button re-triggers file picker; Remove button clears only that band.
  - File name and formatted size (KB/MB) are displayed.
  - `Run Cloud Segmentation` button is strictly disabled until all 4 bands are loaded.
  - Submitting sends valid Form-Data matching the backend contract.

### Stage 3: Scientific Results Dashboard
- **Target Files to Create:**
  - `frontend/src/components/results/MetricsSummary.jsx` (cloud %, clear sky %, pixel counts, latency).
  - `frontend/src/components/results/ImageStage.jsx` (tabs for Overlay, Binary Mask, RGB, Heatmap; before/after split slider; fullscreen modal; PNG download).
  - `frontend/src/components/results/TelemetryCard.jsx` (device, tensor dimensions, model epoch, loss).
  - `frontend/src/components/results/ResultsDashboard.jsx` (dashboard view orchestrator).
- **Dependencies:** Stage 2 complete.
- **Acceptance Gate:**
  - All real backend metrics display accurately.
  - Layer tabs switch seamlessly between all 4 base64 images.
  - Split comparison slider allows dragging horizontally over imagery.
  - Fullscreen lightbox expands images clearly.
  - PNG export triggers a genuine file download for each active layer.
  - `Analyze New Scene` button returns cleanly to upload workspace.

### Stage 4: App Integration, Verification & Polish
- **Target Files to Modify:**
  - `frontend/src/App.jsx` (point view rendering to new `LandingView`, `UploadWorkspace`, and `ResultsDashboard`).
  - `frontend/src/index.css` (import `tokens.css` and set base root styles).
- **Dependencies:** Stages 1–3 complete.
- **Acceptance Gate:**
  - `npm run lint` passes with 0 warnings and 0 errors.
  - `npm run build` succeeds with zero errors.
  - End-to-end user journey works smoothly: Landing -> Upload -> Inference -> Results -> New Scene.
  - No memory leaks or canvas context loss when switching views back and forth.

---

## 8. Acceptance Criteria & Regression Verification

### Functional Verification Checklist
1. **Landing:**
   - [ ] Exactly one primary CTA button exists, labeled `Start Analysis`.
   - [ ] No fake telemetry cards, scrolling tickers, status badges, or metric widgets on the landing page.
   - [ ] Earth rotates continuously around a tilted axis.
   - [ ] Dragging horizontally and vertically rotates the Earth.
   - [ ] Automatic rotation resumes smoothly after pointer release.
   - [ ] Four satellites orbit visibly on separate orbital planes.
2. **Upload:**
   - [ ] Four band cards present: Red (B04), Green (B03), Blue (B02), NIR (B08).
   - [ ] Each card supports drag-and-drop and click-to-browse.
   - [ ] Each card has an individual Replace action and an individual Remove action.
   - [ ] Selected files display original filename and formatted file size.
   - [ ] Threshold slider (default 50%) and Opacity slider (default 45%) function properly.
   - [ ] `Run Cloud Segmentation` button is disabled until all 4 bands are loaded.
3. **Results:**
   - [ ] Cloud coverage percentage and clear sky percentage match backend response.
   - [ ] Cloud, clear, and total pixel counts format with comma separators.
   - [ ] Inference latency displays backend timing in milliseconds.
   - [ ] All 4 visual layers (Overlay, Binary Mask, RGB Composite, Heatmap) display properly.
   - [ ] Split-slider allows interactive before/after image inspection.
   - [ ] Fullscreen modal expands image on click.
   - [ ] PNG download button triggers actual browser download.
4. **Code Quality:**
   - [ ] `npm run lint` returns 0 warnings and 0 errors.
   - [ ] `npm run build` compiles clean production bundle.
   - [ ] No modifications to backend or ML code.

---

## 9. Safe, Change-Aware Git Workflow & Recovery Guidance

### Strict Prohibition Against Destructive Git Commands
- **NEVER execute destructive working-tree commands** without explicit user instruction. Specifically:
  - Do NOT run `git checkout -- <path>` or `git restore <path>`.
  - Do NOT run `git reset --hard`.
  - Do NOT run `git clean -fd`.
- Destructive commands risk permanently destroying uncommitted user work and scratch notes.

### Safe Recovery Workflow
1. **Always Inspect First:**
   ```bash
   git status
   git diff
   ```
2. **Safe Stashing (Preserves All Changes):**
   If an experimental change needs to be temporarily set aside without data loss, safely stash it:
   ```bash
   git stash push -m "WIP before investigating issue"
   ```
   To restore stashed work later:
   ```bash
   git stash list
   git stash pop
   ```
3. **Non-Destructive Component Fallback:**
   Because all new components are created in isolated subdirectories (`src/components/common/`, `3d/`, `landing/`, `upload/`, `results/`), the original working files (`LandingPage.jsx`, `ProcessingDashboard.jsx`, `ResultViewer.jsx`) remain 100% intact. Reverting to the baseline UI simply requires pointing the import lines in `App.jsx` back to the original files, with zero risk of code loss.
4. **Branch Protection:**
   All work remains strictly isolated on `feat/immersive-earth-frontend`. No commits or pushes should occur without explicit user approval.
