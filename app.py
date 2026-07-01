"""
app.py – Premium Streamlit Dashboard for the ISRO IR Colorization Pipeline
===========================================================================
Run:  streamlit run app.py
"""

import os
import sys
import json
import time
import threading
import io
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from models import SuperResolutionNet, SegmentationNet, ColorizationNet
from dataset import normalize_tir, normalize_rgb, IRPatchDataset
from pipeline import run_inference, SEG_CLASS_COLORS, SEG_CLASS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISRO IR Colorization – BAH 2026",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark space theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
    color: #e0e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b35 0%, #0a1525 100%);
    border-right: 1px solid #1e3a5f;
}

/* Header Banner */
.hero-banner {
    background: linear-gradient(135deg, #0f2044 0%, #1a3a6e 40%, #0e4d8a 70%, #0d3060 100%);
    border: 1px solid #2a5298;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(41,128,185,0.2) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #64b5f6 0%, #81d4fa 40%, #e1f5fe 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: #90caf9;
    font-size: 1.05rem;
    margin-top: 8px;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(41,128,185,0.25);
    border: 1px solid #2980b9;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #81d4fa;
    margin-right: 8px;
    margin-top: 12px;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #0f2044 0%, #132952 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(41,128,185,0.2);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #64b5f6;
    line-height: 1;
}
.metric-label {
    font-size: 0.85rem;
    color: #78909c;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Stage pipeline */
.pipeline-stage {
    background: linear-gradient(135deg, #0d1b35 0%, #132952 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid #2980b9;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #64b5f6;
    padding: 8px 0;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 16px;
}

/* Image caption */
.img-caption {
    text-align: center;
    font-size: 0.8rem;
    color: #546e7a;
    margin-top: 4px;
}

/* Progress bar color override */
.stProgress > div > div { background-color: #2980b9; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1b35;
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #78909c;
    background: transparent;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    color: #64b5f6 !important;
    background: rgba(41,128,185,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
    box-shadow: 0 4px 20px rgba(41,128,185,0.4);
    transform: translateY(-1px);
}

/* Slider track */
.stSlider [data-baseweb="slider"] { color: #2980b9; }

hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
for key, val in {
    'models_loaded': False,
    'sr_model': None, 'seg_model': None, 'col_model': None,
    'inference_results': None,
    'train_log': [],
    'training': False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR   = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
PATCHES_DIR = os.path.join(BASE_DIR, 'output', 'patches')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output', 'model_outputs')


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models_cached(weights_dir):
    """Load all three models (cached across reruns)."""
    def _load(cls, fname):
        m = cls().to(DEVICE)
        path = os.path.join(weights_dir, fname)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        return m

    sr  = _load(SuperResolutionNet, 'sr_best.pth')
    seg = _load(SegmentationNet,    'seg_best.pth')
    col = _load(ColorizationNet,    'col_best.pth')
    return sr, seg, col


def npy_to_display(arr: np.ndarray, mode='tir') -> np.ndarray:
    """Convert raw numpy array to uint8 display image."""
    if mode == 'tir':
        norm = normalize_tir(arr.squeeze())
        u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)  # heat colormap (BGR)
    elif mode == 'rgb':
        arr = np.clip(arr, 0, 1)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def arr_to_pil(arr: np.ndarray, bgr=False) -> Image.Image:
    if bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr.astype(np.uint8))


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255.0**2 / mse)


def compute_ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    if pred.ndim == 3:
        return ssim(pred, target, channel_axis=-1, data_range=255)
    return ssim(pred, target, data_range=255)


def load_history(name):
    path = os.path.join(WEIGHTS_DIR, f'{name}_history.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plotly_dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,53,0.6)',
        font=dict(color='#90caf9', family='Inter'),
        xaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f'),
        yaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#90caf9')),
        margin=dict(l=40, r=20, t=40, b=40),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0;'>
        <div style='font-size:2.5rem;'>🛰️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#64b5f6;'>IR Colorization</div>
        <div style='font-size:0.75rem; color:#546e7a;'>BAH 2026 Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Device info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.markdown(f"**🖥️ Device:** `{gpu_name}`")
    st.markdown(f"**📁 Weights:** `{os.path.basename(WEIGHTS_DIR)}/`")

    st.divider()

    # Navigation
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 Inference",
        "🏋️ Train Models",
        "📊 Metrics & History",
        "📁 Data Explorer",
    ], label_visibility="collapsed")

    st.divider()

    # Load models button
    if st.button("⚡ Load / Reload Models", use_container_width=True):
        with st.spinner("Loading models…"):
            try:
                sr, seg, col = load_models_cached(WEIGHTS_DIR)
                st.session_state.sr_model  = sr
                st.session_state.seg_model = seg
                st.session_state.col_model = col
                st.session_state.models_loaded = True
                st.success("Models loaded ✓")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.models_loaded:
        st.markdown("✅ **Models ready**")
    else:
        st.markdown("⚠️ **Models not loaded**")


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER (all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🛰️ Infrared Image Colorization & Enhancement</h1>
    <p class="hero-sub">Bhartiya Antriksh Hackathon 2026 · ISRO · Landsat 9 TIR Pipeline</p>
    <span class="badge">Super-Resolution</span>
    <span class="badge">Segmentation</span>
    <span class="badge">Colorization</span>
    <span class="badge">PyTorch</span>
    <span class="badge">RTX 3050</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)

    # Pipeline stages
    stages = [
        ("Stage 1 – Super-Resolution", "📡",
         "SRCNN with residual blocks. Upscales 200m TIR → 100m TIR (2× upscaling via sub-pixel convolution). Input: (1,256,256), Output: (1,512,512)."),
        ("Stage 2 – Semantic Segmentation", "🗺️",
         "Lightweight U-Net predicting 6-class land-cover masks (Water, Vegetation, Agriculture, Barren, Urban, Industrial). Input: (1,512,512)."),
        ("Stage 3 – Semantic Colorization", "🎨",
         "Semantic-guided U-Net. Takes concatenated TIR + segmentation features → synthesizes RGB. Input: (7,512,512), Output: (3,512,512) RGB."),
    ]

    for title, icon, desc in stages:
        st.markdown(f"""
        <div class="pipeline-stage">
            <strong style='color:#64b5f6'>{icon} {title}</strong><br>
            <span style='color:#90caf9; font-size:0.9rem'>{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Data Flow</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    steps = [
        ("200m TIR B10", "Raw satellite band"),
        ("SR Model", "→ 100m TIR"),
        ("Seg Model", "→ Class Mask"),
        ("Color Model", "→ RGB"),
        ("Output TIF", "Submission Ready"),
    ]
    for col, (title, sub) in zip([col1, col2, col3, col4, col5], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{sub}</div>
                <div style='font-size:0.95rem; color:#e0e8f0; margin-top:6px; font-weight:600'>{title}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Segmentation Classes</div>', unsafe_allow_html=True)

    class_cols = st.columns(6)
    class_info = [
        ("Water",        "#0000ff", "Rivers, lakes"),
        ("Vegetation",   "#008000", "Forests, grass"),
        ("Agriculture",  "#00ff00", "Crop fields"),
        ("Barren",       "#804000", "Dry land, rock"),
        ("Urban",        "#ff0000", "Cities, roads"),
        ("Hot/Industr.", "#ffa500", "Industrial heat"),
    ]
    for col, (name, color, desc) in zip(class_cols, class_info):
        with col:
            st.markdown(f"""
            <div style='background:{color}22; border:1px solid {color}55; border-radius:8px;
                        padding:10px; text-align:center;'>
                <div style='width:20px;height:20px;background:{color};border-radius:50%;
                            margin:0 auto 6px;'></div>
                <div style='font-size:0.85rem;font-weight:600;color:#e0e8f0'>{name}</div>
                <div style='font-size:0.7rem;color:#78909c'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Check for existing weights
    st.divider()
    st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
    mcols = st.columns(3)
    for col, (name, fname) in zip(mcols, [
        ("Super-Resolution", "sr_best.pth"),
        ("Segmentation",     "seg_best.pth"),
        ("Colorization",     "col_best.pth"),
    ]):
        exists = os.path.exists(os.path.join(WEIGHTS_DIR, fname))
        icon = "✅" if exists else "❌"
        color = "#2e7d32" if exists else "#b71c1c"
        status = "Trained" if exists else "Not trained"
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:2rem'>{icon}</div>
                <div class="metric-label">{name}</div>
                <div style='color:{color}; font-size:0.85rem; margin-top:4px'>{status}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Inference":
    st.markdown('<div class="section-header">Run Inference</div>', unsafe_allow_html=True)

    # Auto-load models if not loaded
    if not st.session_state.models_loaded:
        with st.spinner("Auto-loading models…"):
            sr, seg, col = load_models_cached(WEIGHTS_DIR)
            st.session_state.sr_model  = sr
            st.session_state.seg_model = seg
            st.session_state.col_model = col
            st.session_state.models_loaded = True

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("📂 Input Selection")

        # Find available samples
        sample_dirs = sorted(Path(PATCHES_DIR).rglob('tir_200m.npy'))
        if not sample_dirs:
            st.warning("No patches found. Run `python driver.py` first or add data.")
            st.stop()

        sample_labels = [str(p.parent.relative_to(BASE_DIR)) for p in sample_dirs]
        selected_label = st.selectbox("Select patch sample", sample_labels,
                                       help="Choose a sample from output/patches")
        selected_path = str(sample_dirs[sample_labels.index(selected_label)].parent)

        st.markdown("**Or upload a custom .npy file:**")
        uploaded = st.file_uploader("Upload TIR 200m .npy", type=['npy'])

        st.markdown("---")
        st.markdown("**🎛️ Display Settings**")
        contrast = st.slider("TIR contrast stretch", 0.0, 10.0, 2.0, 0.5)

        run_btn = st.button("🚀 Run Pipeline", use_container_width=True)

    if run_btn:
        try:
            if uploaded:
                tir200_arr = np.load(io.BytesIO(uploaded.read())).astype(np.float32)
            else:
                tir200_arr = np.load(os.path.join(selected_path, 'tir_200m.npy')).astype(np.float32)
            if tir200_arr.ndim == 3:
                tir200_arr = tir200_arr[0]

            with st.spinner("⚙️ Running 3-stage pipeline…"):
                t0 = time.time()
                results = run_inference(
                    tir200_arr,
                    st.session_state.sr_model,
                    st.session_state.seg_model,
                    st.session_state.col_model,
                    DEVICE,
                )
                elapsed = time.time() - t0
                st.session_state.inference_results = {
                    'results': results,
                    'elapsed': elapsed,
                    'sample_path': selected_path,
                }

            # Load GT if available for metrics
            gt_path = os.path.join(selected_path, 'rgb_100m_512.npy')
            if os.path.exists(gt_path):
                gt_rgb = np.load(gt_path).astype(np.float32)
                if gt_rgb.ndim == 3 and gt_rgb.shape[0] == 3:
                    gt_rgb = gt_rgb.transpose(1, 2, 0)
                gt_rgb = normalize_rgb(gt_rgb)
                pred_rgb_u8 = (np.clip(results['rgb_pred'], 0, 1) * 255).astype(np.uint8)
                gt_rgb_u8   = (np.clip(gt_rgb[:,:,:3], 0, 1) * 255).astype(np.uint8)
                psnr_val = compute_psnr(pred_rgb_u8, gt_rgb_u8)
                st.session_state.inference_results['gt_rgb'] = gt_rgb
                st.session_state.inference_results['psnr']   = psnr_val

        except Exception as e:
            st.error(f"Inference failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results
    if st.session_state.inference_results:
        res     = st.session_state.inference_results['results']
        elapsed = st.session_state.inference_results['elapsed']

        st.markdown(f"<div style='color:#2e7d32;font-size:1.1rem;font-weight:600'>✅ Inference completed in {elapsed:.2f}s</div>",
                    unsafe_allow_html=True)

        # Metrics row
        psnr_val = st.session_state.inference_results.get('psnr', None)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("""<div class="metric-card"><div class="metric-value">200m</div>
            <div class="metric-label">Input TIR Res</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown("""<div class="metric-card"><div class="metric-value">100m</div>
            <div class="metric-label">SR Output Res</div></div>""", unsafe_allow_html=True)
        with m3:
            psnr_str = f"{psnr_val:.1f}dB" if psnr_val else "N/A"
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{psnr_str}</div>
            <div class="metric-label">PSNR vs GT</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{elapsed:.2f}s</div>
            <div class="metric-label">Inference Time</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # 4-panel visualization
        st.markdown('<div class="section-header">Pipeline Outputs</div>', unsafe_allow_html=True)
        v1, v2, v3, v4 = st.columns(4)

        panels = [
            (v1, "📡 Input TIR 200m",  npy_to_display(res['tir_200m_norm'], 'tir'), True),
            (v2, "🔭 SR TIR 100m",     npy_to_display(res['tir_100m_sr'],   'tir'), True),
            (v3, "🗺️ Segmentation",    res['seg_color'],                           False),
            (v4, "🎨 Colorized RGB",    (np.clip(res['rgb_pred'],0,1)*255).astype(np.uint8), False),
        ]
        for vcol, title, img_arr, is_bgr in panels:
            with vcol:
                st.markdown(f"**{title}**")
                pil_img = arr_to_pil(img_arr, bgr=is_bgr)
                st.image(pil_img, use_container_width=True)
                st.markdown(f"<p class='img-caption'>{img_arr.shape[1]}×{img_arr.shape[0]}</p>",
                            unsafe_allow_html=True)

        # Histogram comparison
        st.markdown("---")
        st.markdown('<div class="section-header">Intensity Histograms</div>', unsafe_allow_html=True)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["TIR 200m Histogram", "SR 100m Histogram"])
        for col_idx, (data, label) in enumerate([
            (res['tir_200m_norm'].flatten(), "TIR 200m"),
            (res['tir_100m_sr'].flatten(),   "TIR SR 100m"),
        ], start=1):
            counts, bins = np.histogram(data, bins=64)
            fig.add_trace(
                go.Bar(x=bins[:-1], y=counts, name=label,
                       marker_color='#2980b9' if col_idx == 1 else '#e74c3c',
                       opacity=0.8),
                row=1, col=col_idx
            )
        fig.update_layout(**plotly_dark_layout(), showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)

        # Segmentation class distribution
        st.markdown('<div class="section-header">Segmentation Class Distribution</div>', unsafe_allow_html=True)
        mask = res['seg_mask']
        counts = [(SEG_CLASS_NAMES[i], int((mask == i).sum())) for i in range(6)]
        counts_sorted = sorted(counts, key=lambda x: -x[1])
        fig2 = go.Figure(go.Bar(
            x=[c[0] for c in counts_sorted],
            y=[c[1] for c in counts_sorted],
            marker_color=['#0000cc','#006600','#00cc00','#663300','#cc0000','#ff8800'],
            opacity=0.85,
        ))
        fig2.update_layout(**plotly_dark_layout(), height=280,
                           xaxis_title="Class", yaxis_title="Pixel Count")
        st.plotly_chart(fig2, use_container_width=True)

        # Download section
        st.markdown("---")
        st.subheader("💾 Export Results")
        d1, d2, d3 = st.columns(3)
        with d1:
            tir_sr_u8 = (np.clip(res['tir_100m_sr'], 0, 1) * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(tir_sr_u8).save(buf, format='PNG')
            st.download_button("⬇️ Download SR TIR PNG", buf.getvalue(), "sr_tir_100m.png", "image/png")
        with d2:
            rgb_u8 = (np.clip(res['rgb_pred'], 0, 1) * 255).astype(np.uint8)
            buf2 = io.BytesIO()
            Image.fromarray(rgb_u8).save(buf2, format='PNG')
            st.download_button("⬇️ Download Colorized PNG", buf2.getvalue(), "colorized_rgb.png", "image/png")
        with d3:
            buf3 = io.BytesIO()
            Image.fromarray(res['seg_color']).save(buf3, format='PNG')
            st.download_button("⬇️ Download Seg Mask PNG", buf3.getvalue(), "seg_mask.png", "image/png")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏋️ Train Models":
    st.markdown('<div class="section-header">Train Models</div>', unsafe_allow_html=True)
    st.info("Training will use patches in `output/patches/`. Weights will be saved to `weights/`.")

    t1, t2, t3 = st.columns(3)
    with t1:
        model_choice = st.selectbox("Select Model", ["All (sr + seg + col)", "Super-Resolution (sr)",
                                                       "Segmentation (seg)", "Colorization (col)"])
    with t2:
        epochs = st.slider("Epochs", 5, 200, 30, 5)
    with t3:
        lr = st.select_slider("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)

    batch_size = st.slider("Batch Size", 1, 8, 2)

    st.markdown("---")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        train_btn = st.button("🏋️ Start Training", use_container_width=True,
                               disabled=st.session_state.training)
    with col_b:
        if st.session_state.training:
            st.markdown("⚙️ **Training in progress…**")

    if train_btn and not st.session_state.training:
        model_map = {
            "All (sr + seg + col)": "all",
            "Super-Resolution (sr)": "sr",
            "Segmentation (seg)": "seg",
            "Colorization (col)": "col",
        }
        model_arg = model_map[model_choice]
        st.session_state.training = True
        st.session_state.train_log = []

        cmd = (
            f"python train.py --model {model_arg} "
            f"--patches_dir {PATCHES_DIR} "
            f"--weights_dir {WEIGHTS_DIR} "
            f"--epochs {epochs} --batch_size {batch_size} --lr {lr}"
        )
        st.code(cmd, language='bash')
        st.warning("⚠️ Run the above command in a terminal to train. The dashboard will display history once done.")
        st.session_state.training = False

    # Show training history from JSON files
    st.markdown("---")
    st.markdown('<div class="section-header">Training History</div>', unsafe_allow_html=True)

    hist_tabs = st.tabs(["Super-Resolution", "Segmentation", "Colorization"])
    for tab, model_key in zip(hist_tabs, ['sr', 'seg', 'col']):
        with tab:
            hist = load_history(model_key)
            if hist is None:
                st.info(f"No training history for {model_key}. Train the model first.")
            else:
                epochs_range = list(range(1, len(hist['train_loss']) + 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs_range, y=hist['train_loss'],
                                         mode='lines+markers', name='Train Loss',
                                         line=dict(color='#e74c3c', width=2)))
                fig.add_trace(go.Scatter(x=epochs_range, y=hist['val_loss'],
                                         mode='lines+markers', name='Val Loss',
                                         line=dict(color='#3498db', width=2)))
                fig.update_layout(**plotly_dark_layout(), height=300,
                                   title=f"{model_key.upper()} Loss Curves",
                                   xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig, use_container_width=True)

                # PSNR or Accuracy
                if 'val_psnr' in hist:
                    fig2 = go.Figure(go.Scatter(x=epochs_range, y=hist['val_psnr'],
                                                mode='lines+markers',
                                                line=dict(color='#2ecc71', width=2)))
                    fig2.update_layout(**plotly_dark_layout(), height=250,
                                       title="Validation PSNR (dB)",
                                       xaxis_title="Epoch", yaxis_title="PSNR (dB)")
                    st.plotly_chart(fig2, use_container_width=True)
                elif 'val_acc' in hist:
                    fig2 = go.Figure(go.Scatter(x=epochs_range, y=hist['val_acc'],
                                                mode='lines+markers',
                                                line=dict(color='#f39c12', width=2)))
                    fig2.update_layout(**plotly_dark_layout(), height=250,
                                       title="Validation Accuracy",
                                       xaxis_title="Epoch", yaxis_title="Accuracy")
                    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: METRICS & HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Metrics & History":
    st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        with st.spinner("Loading models…"):
            sr, seg, col = load_models_cached(WEIGHTS_DIR)
            st.session_state.sr_model  = sr
            st.session_state.seg_model = seg
            st.session_state.col_model = col
            st.session_state.models_loaded = True

    eval_btn = st.button("📐 Evaluate on All Demo Patches", use_container_width=False)

    if eval_btn:
        sample_paths = sorted(Path(PATCHES_DIR).rglob('tir_200m.npy'))
        if not sample_paths:
            st.warning("No patches found.")
        else:
            psnr_list, ssim_list, times = [], [], []
            progress = st.progress(0)
            status   = st.empty()

            for i, npy_path in enumerate(sample_paths):
                sample_dir = str(npy_path.parent)
                tir200 = np.load(str(npy_path)).astype(np.float32)
                if tir200.ndim == 3:
                    tir200 = tir200[0]

                t0 = time.time()
                res = run_inference(tir200,
                                    st.session_state.sr_model,
                                    st.session_state.seg_model,
                                    st.session_state.col_model,
                                    DEVICE)
                times.append(time.time() - t0)

                gt_path = os.path.join(sample_dir, 'rgb_100m_512.npy')
                if os.path.exists(gt_path):
                    gt = np.load(gt_path).astype(np.float32)
                    if gt.ndim == 3 and gt.shape[0] == 3:
                        gt = gt.transpose(1, 2, 0)
                    gt = normalize_rgb(gt)
                    pred_u8 = (np.clip(res['rgb_pred'], 0, 1) * 255).astype(np.uint8)
                    gt_u8   = (np.clip(gt[:,:,:3], 0, 1) * 255).astype(np.uint8)
                    psnr_list.append(compute_psnr(pred_u8, gt_u8))
                    try:
                        ssim_list.append(compute_ssim_simple(pred_u8, gt_u8))
                    except Exception:
                        pass

                progress.progress((i + 1) / len(sample_paths))
                status.markdown(f"Processed {i+1}/{len(sample_paths)} samples…")

            st.session_state['eval_metrics'] = {
                'psnr': psnr_list, 'ssim': ssim_list, 'times': times
            }

    metrics = st.session_state.get('eval_metrics')
    if metrics:
        psnr_list = metrics['psnr']
        ssim_list = metrics['ssim']
        times     = metrics['times']

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            v = f"{np.mean(psnr_list):.2f} dB" if psnr_list else "N/A"
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{v}</div>
            <div class="metric-label">Avg PSNR</div></div>""", unsafe_allow_html=True)
        with mc2:
            v2 = f"{np.mean(ssim_list):.4f}" if ssim_list else "N/A"
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{v2}</div>
            <div class="metric-label">Avg SSIM</div></div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{np.mean(times):.2f}s</div>
            <div class="metric-label">Avg Inference</div></div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(times)}</div>
            <div class="metric-label">Samples Eval'd</div></div>""", unsafe_allow_html=True)

        if psnr_list:
            fig = go.Figure(go.Bar(x=list(range(len(psnr_list))), y=psnr_list,
                                   marker_color='#2980b9', name='PSNR'))
            fig.update_layout(**plotly_dark_layout(), height=300,
                               title="Per-Sample PSNR", xaxis_title="Sample", yaxis_title="PSNR (dB)")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Data Explorer":
    st.markdown('<div class="section-header">Dataset Explorer</div>', unsafe_allow_html=True)

    sample_paths = sorted(Path(PATCHES_DIR).rglob('tir_200m.npy'))
    if not sample_paths:
        st.warning("No patch samples found. Run `python driver.py` to generate patches from Landsat data.")
        st.markdown("""
        **Quick start:**
        ```bash
        # Place Landsat 9 bands in input/<product_id>/
        python driver.py
        ```
        """)
    else:
        sample_labels = [str(p.parent.relative_to(BASE_DIR)) for p in sample_paths]
        sel = st.selectbox("Select sample", sample_labels)
        sel_dir = str(sample_paths[sample_labels.index(sel)].parent)

        npy_files = list(Path(sel_dir).glob('*.npy'))
        png_files = list(Path(sel_dir).glob('*.png'))

        col_npy, col_png = st.columns(2)
        with col_npy:
            st.markdown("**NPY Files (training data)**")
            for nf in npy_files:
                arr = np.load(str(nf))
                st.markdown(f"- `{nf.name}` — shape: `{arr.shape}` dtype: `{arr.dtype}` "
                            f"min: `{arr.min():.2f}` max: `{arr.max():.2f}`")

        with col_png:
            st.markdown("**PNG Files (previews)**")
            for pf in png_files:
                img = Image.open(str(pf))
                st.image(img, caption=pf.name, use_container_width=True)

        # Detailed view
        st.markdown("---")
        st.markdown('<div class="section-header">Band Statistics</div>', unsafe_allow_html=True)

        for nf in npy_files:
            arr = np.load(str(nf)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            arr_flat = arr.flatten()

            exp = st.expander(f"📊 {nf.name}")
            with exp:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min",  f"{arr.min():.2f}")
                c2.metric("Max",  f"{arr.max():.2f}")
                c3.metric("Mean", f"{arr.mean():.2f}")
                c4.metric("Std",  f"{arr.std():.2f}")

                counts, bins = np.histogram(arr_flat[~np.isnan(arr_flat)], bins=64)
                fig = go.Figure(go.Bar(x=bins[:-1], y=counts, marker_color='#2980b9', opacity=0.8))
                fig.update_layout(**plotly_dark_layout(), height=200,
                                   margin=dict(l=20, r=10, t=10, b=20))
                st.plotly_chart(fig, use_container_width=True)
