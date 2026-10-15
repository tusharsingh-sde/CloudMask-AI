import streamlit as st
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import io
import os

from model import UNet

# ─────────────────────── Page Config ───────────────────────
st.set_page_config(
    page_title="CloudMask AI",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── Custom CSS ───────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #0a0a0a;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #e5e5e5, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #737373;
        font-size: 1.1rem;
    }
    
    /* Stat cards */
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #d4d4d4, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #737373;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    
    /* Upload section */
    .upload-header {
        color: #a3a3a3;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111111;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .status-error {
        background: rgba(239,68,68,0.15);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Load Model (Cached) ───────────────────────
@st.cache_resource
def load_model():
    """Load the trained UNet model once and cache it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=4, out_channels=1).to(device)
    
    model_path = "cloudmask_unet_best.pt"
    if not os.path.exists(model_path):
        return None, device, None
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device, checkpoint

model, device, checkpoint = load_model()

# ─────────────────────── Header ───────────────────────
st.markdown("""
<div class="main-header">
    <h1>☁️ CloudMask AI</h1>
    <p>Upload satellite band images to detect cloud coverage using deep learning</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────── Sidebar ───────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Info")
    
    if model is not None:
        st.markdown('<span class="status-badge status-ready">✅ Model Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"**Device:** `{device.upper()}`")
        if checkpoint:
            st.markdown(f"**Trained Epoch:** `{checkpoint.get('epoch', 'N/A')}`")
            best_loss = checkpoint.get('best_loss', None)
            if best_loss is not None:
                st.markdown(f"**Best Loss:** `{best_loss:.4f}`")
    else:
        st.markdown('<span class="status-badge status-error">❌ Model Not Found</span>', unsafe_allow_html=True)
        st.warning("Place `cloudmask_unet_best.pt` in the project root.")
    
    st.markdown("---")
    st.markdown("## 🎛️ Settings")
    confidence_threshold = st.slider("Cloud Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                                     help="Pixels with probability above this are classified as cloud.")
    overlay_opacity = st.slider("Mask Overlay Opacity", 0.1, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.markdown("## 📋 How to Use")
    st.markdown("""
    1. Upload **4 satellite band** images (Red, Green, Blue, NIR)
    2. Click **🚀 Run Prediction**
    3. View the cloud mask & coverage stats
    
    > Images should be from the **Landsat 8** 38-Cloud dataset format (`.TIF` or `.jpg`)
    """)

# ─────────────────────── Helper Functions ───────────────────────
def read_uploaded_image(uploaded_file):
    """Read an uploaded file into a grayscale numpy array."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img

def create_rgb_composite(r, g, b):
    """Create a normalized RGB composite for display."""
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    # Normalize to 0-255 for display
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
    return rgb

def create_overlay(rgb, mask, opacity=0.45):
    """Overlay the cloud mask on the RGB composite."""
    overlay = rgb.copy().astype(np.float32)
    # Cloud pixels → semi-transparent red/magenta
    cloud_color = np.array([255, 50, 120], dtype=np.float32)
    cloud_mask_3ch = np.stack([mask, mask, mask], axis=-1)
    overlay = np.where(cloud_mask_3ch > 0, 
                       overlay * (1 - opacity) + cloud_color * opacity, 
                       overlay)
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ─────────────────────── Upload Section ───────────────────────
st.markdown("### 📤 Upload Satellite Band Images")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="upload-header">🔴 Red Band</p>', unsafe_allow_html=True)
    red_file = st.file_uploader("Red", type=["tif", "tiff", "jpg", "png"], label_visibility="collapsed", key="red")

with col2:
    st.markdown('<p class="upload-header">🟢 Green Band</p>', unsafe_allow_html=True)
    green_file = st.file_uploader("Green", type=["tif", "tiff", "jpg", "png"], label_visibility="collapsed", key="green")

with col3:
    st.markdown('<p class="upload-header">🔵 Blue Band</p>', unsafe_allow_html=True)
    blue_file = st.file_uploader("Blue", type=["tif", "tiff", "jpg", "png"], label_visibility="collapsed", key="blue")

with col4:
    st.markdown('<p class="upload-header">🟣 NIR Band</p>', unsafe_allow_html=True)
    nir_file = st.file_uploader("NIR", type=["tif", "tiff", "jpg", "png"], label_visibility="collapsed", key="nir")

# ─────────────────────── Prediction ───────────────────────
all_uploaded = all([red_file, green_file, blue_file, nir_file])

if all_uploaded:
    st.markdown("---")
    
    # Read images
    r = read_uploaded_image(red_file)
    g = read_uploaded_image(green_file)
    b = read_uploaded_image(blue_file)
    n = read_uploaded_image(nir_file)
    
    if any(x is None for x in [r, g, b, n]):
        st.error("❌ One or more images could not be read. Please check the file format.")
    else:
        # Show input band previews
        st.markdown("### 🛰️ Input Band Previews")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.image(r, caption=f"Red ({r.shape[1]}×{r.shape[0]})", use_container_width=True, clamp=True)
        with pc2:
            st.image(g, caption=f"Green ({g.shape[1]}×{g.shape[0]})", use_container_width=True, clamp=True)
        with pc3:
            st.image(b, caption=f"Blue ({b.shape[1]}×{b.shape[0]})", use_container_width=True, clamp=True)
        with pc4:
            st.image(n, caption=f"NIR ({n.shape[1]}×{n.shape[0]})", use_container_width=True, clamp=True)
        
        st.markdown("---")
        
        # Run prediction button
        if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
            if model is None:
                st.error("❌ Model not loaded! Place `cloudmask_unet_best.pt` in the project root.")
            else:
                with st.spinner("🔮 Running inference..."):
                    # Stack and convert to tensor
                    img_stacked = np.stack([r, g, b, n], axis=-1)
                    image_tensor = TF.to_tensor(img_stacked).unsqueeze(0).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        output = model(image_tensor)
                        probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                        predicted_mask = (probabilities > confidence_threshold).astype(np.float32)
                    
                    # Stats
                    total_pixels = predicted_mask.size
                    cloud_pixels = int(np.count_nonzero(predicted_mask))
                    clear_pixels = total_pixels - cloud_pixels
                    cloud_pct = (cloud_pixels / total_pixels) * 100
                    clear_pct = 100 - cloud_pct
                
                # ─── Results ───
                st.markdown("### 📊 Results")
                
                # Stat cards
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{cloud_pct:.1f}%</div>
                        <div class="stat-label">☁️ Cloud Coverage</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{clear_pct:.1f}%</div>
                        <div class="stat-label">🌤️ Clear Sky</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{total_pixels:,}</div>
                        <div class="stat-label">📐 Total Pixels</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visual outputs
                rgb_composite = create_rgb_composite(r, g, b)
                mask_display = (predicted_mask * 255).astype(np.uint8)
                overlay_img = create_overlay(rgb_composite, predicted_mask, opacity=overlay_opacity)
                
                v1, v2, v3 = st.columns(3)
                with v1:
                    st.image(rgb_composite, caption="🛰️ RGB Composite", use_container_width=True)
                with v2:
                    st.image(mask_display, caption="🎭 Predicted Cloud Mask", use_container_width=True)
                with v3:
                    st.image(overlay_img, caption="🔍 Mask Overlay", use_container_width=True)
                
                # Probability heatmap
                st.markdown("### 🌡️ Cloud Probability Heatmap")
                prob_heatmap = cv2.applyColorMap((probabilities * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                prob_heatmap = cv2.cvtColor(prob_heatmap, cv2.COLOR_BGR2RGB)
                st.image(prob_heatmap, caption=f"Probability Map (Threshold: {confidence_threshold})", use_container_width=True)
                
                st.success(f"✅ Prediction complete! Cloud coverage: **{cloud_pct:.2f}%**")

elif any([red_file, green_file, blue_file, nir_file]):
    # Some but not all uploaded
    missing = []
    if not red_file: missing.append("Red")
    if not green_file: missing.append("Green")
    if not blue_file: missing.append("Blue")
    if not nir_file: missing.append("NIR")
    st.info(f"⏳ Still waiting for: **{', '.join(missing)}** band(s)")
else:
    # Nothing uploaded - show placeholder
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; background:rgba(255,255,255,0.03); 
                border-radius:16px; border:1px dashed rgba(255,255,255,0.15); margin-top:1rem;">
        <p style="font-size:3rem; margin-bottom:0.5rem;">🛰️</p>
        <p style="color:#d4d4d4; font-size:1.2rem; font-weight:600;">Upload 4 satellite band images to get started</p>
        <p style="color:#737373; font-size:0.9rem;">Red • Green • Blue • NIR</p>
    </div>
    """, unsafe_allow_html=True)
