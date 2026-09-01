# =============================================================================
# app.py
# Brain Tumor Detection - Professional Streamlit Web Application
# =============================================================================
# Run with:  streamlit run app.py
#
# Features:
#   - Modern medical-themed dark UI
#   - Drag-and-drop MRI image upload
#   - Real-time tumor detection with ViT-GRU
#   - Grad-CAM explainability heatmaps
#   - Prediction confidence chart
#   - Downloadable PDF report
#   - Multiple image analysis
#   - Responsive sidebar navigation
# =============================================================================

import os
import io
import base64
import datetime
import logging
import traceback
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import streamlit as st
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu

# Local modules
from predict import load_model, predict_single
from explainability import BrainTumorGradCAM, create_explainability_figure
from dataset import CLASS_NAMES, IDX_TO_CLASS, get_inference_transforms

# =============================================================================
# CONFIGURATION
# =============================================================================

# Page must be configured FIRST before any other Streamlit calls
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/BrainTumorViT_GRU",
        "About": "Brain Tumor Detection using Hybrid ViT-GRU with Explainable AI",
    },
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuroScanApp")

# Constants
CHECKPOINT_PATH  = "saved_models/best_model.pth"
DEMO_IMAGE_DIR   = "uploads/"
OUTPUT_DIR       = "outputs/"
ALLOWED_TYPES    = ["jpg", "jpeg", "png", "bmp", "tiff"]
MAX_FILE_SIZE_MB = 10

os.makedirs(DEMO_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def inject_custom_css() -> None:
    """Inject custom CSS for the medical-grade dark UI theme."""
    st.markdown("""
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2e 0%, #0a1628 100%) !important;
        border-right: 1px solid #1e3a5f;
    }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #0f2340 0%, #0d1b2e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 120, 255, 0.15);
    }

    /* ── Alert boxes ── */
    .alert-danger {
        background: linear-gradient(135deg, #2d0a0a, #1a0505);
        border: 1px solid #e74c3c;
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
        color: #ff8888;
    }
    .alert-success {
        background: linear-gradient(135deg, #0a2d0a, #051a05);
        border: 1px solid #2ecc71;
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
        color: #88ff88;
    }
    .alert-warning {
        background: linear-gradient(135deg, #2d1a00, #1a0f00);
        border: 1px solid #f39c12;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
        color: #ffcc66;
    }
    .alert-info {
        background: linear-gradient(135deg, #001a2d, #000f1a);
        border: 1px solid #3498db;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
        color: #88ccff;
    }

    /* ── Hero header ── */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #81d4fa, #b3e5fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #4fc3f7;
        padding: 8px 0;
        border-bottom: 2px solid #1e3a5f;
        margin-bottom: 1.2rem;
    }

    /* ── Prediction badge ── */
    .pred-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Confidence meter ── */
    .confidence-bar-container {
        background: #0a1628;
        border-radius: 8px;
        height: 12px;
        width: 100%;
        overflow: hidden;
        margin: 6px 0;
    }

    /* ── Upload area ── */
    .upload-area {
        border: 2px dashed #1e3a5f;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        background: rgba(15, 35, 64, 0.4);
        transition: border-color 0.2s;
    }
    .upload-area:hover {
        border-color: #4fc3f7;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
        letter-spacing: 0.03em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);
    }

    /* ── Image containers ── */
    .img-container {
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        overflow: hidden;
        background: #0a1628;
    }

    /* ── Status badge ── */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid #2ecc71;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #2ecc71;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
        margin: 1.5rem 0;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        font-size: 0.75rem;
        color: #64748b;
        background: rgba(10, 22, 40, 0.6);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 1rem;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# CACHED MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_cached_model(checkpoint_path: str, device_str: str):
    """
    Load model once and cache it in Streamlit's resource cache.
    This prevents reloading the model on every user interaction.

    Args:
        checkpoint_path: Path to model checkpoint
        device_str:      Device string ('cuda' or 'cpu')

    Returns:
        Tuple of (model, device) or (None, device) if model not found
    """
    device = torch.device(device_str)

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, device

    try:
        model = load_model(checkpoint_path, device=device)
        return model, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, device


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Convert matplotlib figure to PNG bytes for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#0a0e1a")
    buf.seek(0)
    return buf.getvalue()


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def validate_image(uploaded_file) -> Optional[Image.Image]:
    """
    Validate and load an uploaded image file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        PIL Image if valid, None if invalid
    """
    # Check file size
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB")
        return None

    # Try to open as image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Invalid image file: {e}")
        return None


def get_class_color(class_name: str) -> str:
    """Return hex color for each tumor class."""
    colors = {
        "glioma":     "#e74c3c",
        "meningioma": "#3498db",
        "no_tumor":   "#2ecc71",
        "pituitary":  "#f39c12",
    }
    return colors.get(class_name, "#9b59b6")


def get_class_emoji(class_name: str) -> str:
    """Return emoji icon for each tumor class."""
    emojis = {
        "glioma":     "🔴",
        "meningioma": "🔵",
        "no_tumor":   "🟢",
        "pituitary":  "🟡",
    }
    return emojis.get(class_name, "⚪")


def get_risk_level(class_name: str) -> tuple:
    """Return risk level label and description for each class."""
    levels = {
        "glioma":     ("HIGH RISK",   "Gliomas are aggressive brain tumors arising from glial cells. Immediate specialist consultation is strongly recommended."),
        "meningioma": ("MODERATE",    "Meningiomas are usually benign and slow-growing. Follow-up imaging and specialist evaluation is advised."),
        "no_tumor":   ("NORMAL",      "No tumor detected in this scan. Continue routine monitoring as advised by your physician."),
        "pituitary":  ("MODERATE",    "Pituitary tumors may affect hormone production. Endocrine and neurological evaluation is recommended."),
    }
    return levels.get(class_name, ("UNKNOWN", "Please consult a medical professional for interpretation."))


def create_probability_chart(probabilities: np.ndarray, predicted_class: str) -> plt.Figure:
    """Create a styled horizontal probability bar chart."""
    class_colors = [get_class_color(c) for c in CLASS_NAMES]
    labels = [c.replace("_", " ").title() for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0a1628")

    bars = ax.barh(labels, probabilities * 100, color=class_colors, height=0.55, edgecolor="none")

    # Highlight predicted class
    pred_idx = CLASS_NAMES.index(predicted_class) if predicted_class in CLASS_NAMES else 0
    bars[pred_idx].set_linewidth(2)
    bars[pred_idx].set_edgecolor("white")
    bars[pred_idx].set_linewidth(1.5)

    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{prob*100:.1f}%",
            va="center", ha="left",
            fontsize=9, color="white", fontweight="bold"
        )

    ax.set_xlim(0, 120)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#94a3b8")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", alpha=0.2, color="white")

    plt.tight_layout()
    return fig


def create_gradcam_figure(
    original_img: np.ndarray,
    grayscale_cam: np.ndarray,
    cam_image: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> plt.Figure:
    """Create the 3-panel Grad-CAM visualization figure."""
    fig = plt.figure(figsize=(12, 4))
    fig.patch.set_facecolor("#0a1628")

    axes = fig.subplots(1, 3)

    # Panel 1: Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original MRI", color="white", fontsize=10, pad=8)
    axes[0].axis("off")

    # Panel 2: Heatmap
    axes[1].imshow(grayscale_cam, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Activation Map", color="white", fontsize=10, pad=8)
    axes[1].axis("off")

    # Panel 3: Overlay
    axes[2].imshow(cam_image)
    pred_color = get_class_color(predicted_class)
    axes[2].set_title(
        f"Highlighted Region\n{predicted_class.replace('_',' ').title()} ({confidence*100:.1f}%)",
        color=pred_color, fontsize=9, pad=8, fontweight="bold"
    )
    axes[2].axis("off")

    plt.tight_layout(pad=0.5)
    return fig


def generate_pdf_report(
    image: Image.Image,
    result: dict,
    cam_image: np.ndarray,
    original_img: np.ndarray,
    grayscale_cam: np.ndarray,
) -> bytes:
    """
    Generate a downloadable PDF analysis report.

    Uses matplotlib to compose the report and saves to bytes.
    (Full PDF generation without fpdf2 — uses matplotlib PDF backend)

    Returns:
        PDF bytes for Streamlit download button
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ── Page 1: Cover & Summary ──────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0a1628")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#0a1628")
        ax.axis("off")

        # Header
        ax.text(0.5, 0.95, "🧠 NeuroScan AI", ha="center", va="top",
                fontsize=22, color="#4fc3f7", fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.90, "Brain Tumor Detection Report", ha="center", va="top",
                fontsize=14, color="#94a3b8", transform=ax.transAxes)

        # Date & disclaimer
        date_str = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")
        ax.text(0.5, 0.85, f"Generated: {date_str}", ha="center", va="top",
                fontsize=9, color="#64748b", transform=ax.transAxes)

        # Horizontal divider
        ax.axhline(y=0.83, color="#1e3a5f", linewidth=1.5, transform=ax.transAxes)

        # Prediction result
        pred_class = result["predicted_class"].replace("_", " ").title()
        conf       = result["confidence"] * 100
        risk, desc = get_risk_level(result["predicted_class"])
        pred_color = get_class_color(result["predicted_class"])

        ax.text(0.5, 0.77, "DIAGNOSIS PREDICTION", ha="center", va="top",
                fontsize=11, color="#94a3b8", transform=ax.transAxes)
        ax.text(0.5, 0.72, pred_class, ha="center", va="top",
                fontsize=20, color=pred_color, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.67, f"Confidence: {conf:.1f}%", ha="center", va="top",
                fontsize=13, color="white", transform=ax.transAxes)
        ax.text(0.5, 0.63, f"Risk Level: {risk}", ha="center", va="top",
                fontsize=11, color=pred_color, fontweight="bold", transform=ax.transAxes)

        ax.axhline(y=0.61, color="#1e3a5f", linewidth=1, transform=ax.transAxes)

        # Probabilities table
        ax.text(0.1, 0.58, "Class Probabilities:", va="top",
                fontsize=11, color="#4fc3f7", fontweight="bold", transform=ax.transAxes)
        y_pos = 0.54
        for cls, prob in result["probabilities"].items():
            color = get_class_color(cls)
            ax.text(0.1, y_pos, f"  {cls.replace('_',' ').title()}", va="top",
                    fontsize=10, color="white", transform=ax.transAxes)
            ax.text(0.7, y_pos, f"{prob*100:.2f}%", va="top",
                    fontsize=10, color=color, fontweight="bold", transform=ax.transAxes)
            y_pos -= 0.05

        ax.axhline(y=y_pos - 0.01, color="#1e3a5f", linewidth=1, transform=ax.transAxes)

        # Clinical note
        ax.text(0.1, y_pos - 0.04, "Clinical Note:", va="top",
                fontsize=11, color="#4fc3f7", fontweight="bold", transform=ax.transAxes)
        ax.text(0.1, y_pos - 0.09, desc, va="top", fontsize=9, color="#94a3b8",
                transform=ax.transAxes, wrap=True, multialignment="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f2340", edgecolor="#1e3a5f"))

        # Disclaimer
        disclaimer = (
            "⚠️  DISCLAIMER: This report is generated by an AI system for research purposes only. "
            "It does not constitute a medical diagnosis. Please consult a qualified radiologist "
            "or neurosurgeon for clinical evaluation."
        )
        ax.text(0.5, 0.08, disclaimer, ha="center", va="top",
                fontsize=7.5, color="#64748b", transform=ax.transAxes,
                wrap=True, multialignment="center",
                style="italic")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Explainability Visualization ──────────────────────────
        fig2 = plt.figure(figsize=(8.5, 5))
        fig2.patch.set_facecolor("#0a1628")

        axes2 = fig2.subplots(1, 3)
        axes2[0].imshow(original_img)
        axes2[0].set_title("Original MRI", color="white", fontsize=9)
        axes2[0].axis("off")

        axes2[1].imshow(grayscale_cam, cmap="hot")
        axes2[1].set_title("Grad-CAM Activation", color="white", fontsize=9)
        axes2[1].axis("off")

        axes2[2].imshow(cam_image)
        axes2[2].set_title("Region Highlight", color=pred_color, fontsize=9)
        axes2[2].axis("off")

        plt.suptitle("Explainable AI — Grad-CAM Analysis", color="white", fontsize=11)
        plt.tight_layout()

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# PAGE SECTIONS
# =============================================================================

def render_sidebar(model_loaded: bool) -> str:
    """Render the sidebar navigation and return selected page."""
    with st.sidebar:
        # Logo / Branding
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size: 3rem;">🧠</div>
            <div style="font-family: 'Space Grotesk'; font-size: 1.4rem; font-weight: 700;
                        background: linear-gradient(90deg, #4fc3f7, #81d4fa);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                NeuroScan AI
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                Brain Tumor Detection System
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Model status
        if model_loaded:
            st.markdown("""
            <div class="status-online">
                <span style="width:8px;height:8px;background:#2ecc71;border-radius:50%;display:inline-block;"></span>
                Model Online
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(231,76,60,0.15);border:1px solid #e74c3c;
                        border-radius:20px;padding:4px 12px;font-size:0.8rem;color:#e74c3c;">
                ⚠️ Model Not Loaded
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["🏠 Home", "🔬 Analyze MRI", "📊 About Model", "❓ Help"],
            icons=["house", "activity", "cpu", "question-circle"],
            default_index=0,
            styles={
                "container":        {"background-color": "transparent", "padding": "0"},
                "icon":             {"color": "#4fc3f7", "font-size": "16px"},
                "nav-link":         {"color": "#94a3b8", "font-size": "14px", "margin": "2px 0",
                                     "border-radius": "8px"},
                "nav-link-selected": {"background-color": "#0f2340", "color": "white",
                                       "border-left": "3px solid #4fc3f7"},
            },
        )

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # Supported classes info
        st.markdown("**Detected Classes**")
        class_info = {
            "Glioma":      ("#e74c3c", "High-grade brain tumor"),
            "Meningioma":  ("#3498db", "Meningeal layer tumor"),
            "No Tumor":    ("#2ecc71", "Normal scan"),
            "Pituitary":   ("#f39c12", "Pituitary gland tumor"),
        }
        for cls, (color, desc) in class_info.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin:6px 0;">
                <div style="width:10px;height:10px;background:{color};border-radius:50%;flex-shrink:0;"></div>
                <div>
                    <span style="color:white;font-size:0.85rem;font-weight:600;">{cls}</span>
                    <span style="color:#64748b;font-size:0.75rem;display:block;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # Version info
        st.markdown("""
        <div style="font-size:0.72rem;color:#475569;">
            <b>Model:</b> ViT-Base/16 + GRU<br>
            <b>Framework:</b> PyTorch + timm<br>
            <b>XAI:</b> Grad-CAM<br>
            <b>Version:</b> 1.0.0
        </div>
        """, unsafe_allow_html=True)

    return selected


def render_home_page() -> None:
    """Render the Home landing page."""
    # Hero section
    st.markdown("""
    <div style="padding: 30px 0 20px 0;">
        <div class="hero-title">🧠 NeuroScan AI</div>
        <div class="hero-subtitle">Advanced Brain Tumor Detection & Classification using<br>
        Hybrid Vision Transformer + GRU with Explainable AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🎯", "4 Classes", "Glioma, Meningioma,\nPituitary, No Tumor"),
        ("🤖", "ViT + GRU", "Hybrid transformer\narchitecture"),
        ("🔍", "Grad-CAM", "Explainable AI\nheatmaps"),
        ("📋", "PDF Report", "Downloadable\nanalysis report"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
                <div style="font-size:1rem;font-weight:700;color:#4fc3f7;margin-bottom:4px;">{title}</div>
                <div style="font-size:0.8rem;color:#94a3b8;line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # About the system
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-header'>About the System</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#94a3b8;line-height:1.8;font-size:0.95rem;">
        <p>NeuroScan AI uses a state-of-the-art <b style="color:#4fc3f7">Hybrid Vision Transformer + GRU</b>
        architecture to analyze brain MRI scans and detect tumor types with high accuracy.</p>

        <p>The system processes MRI images through a pretrained <b style="color:#4fc3f7">ViT-Base/16</b> backbone
        that splits the image into 196 patches. These patch embeddings are then fed into a
        <b style="color:#4fc3f7">2-layer GRU</b> for sequential feature learning, followed by a
        classification head.</p>

        <p>Explainability is powered by <b style="color:#4fc3f7">Grad-CAM</b> which generates
        heatmaps highlighting the brain regions most influential to the prediction, making
        the AI decision transparent and interpretable.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header'>Model Architecture</div>", unsafe_allow_html=True)
        arch_steps = [
            ("Input MRI", "224×224×3 image"),
            ("ViT Patches", "196 patch tokens (768-dim)"),
            ("GRU Sequence", "Sequential feature learning"),
            ("FC Layers", "256 → 128 → 4"),
            ("Output", "4 tumor class probabilities"),
        ]
        for step, desc in arch_steps:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin:8px 0;
                        background:#0f2340;border-radius:8px;padding:8px 12px;">
                <div style="width:8px;height:8px;background:#4fc3f7;border-radius:50%;flex-shrink:0;"></div>
                <div>
                    <span style="color:white;font-weight:600;font-size:0.9rem;">{step}</span>
                    <span style="color:#64748b;font-size:0.8rem;"> — {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This tool is designed for research and educational purposes only.
    It should NOT be used as a substitute for professional medical diagnosis. Always consult a
    qualified radiologist or neurosurgeon for clinical evaluation and treatment decisions.
    </div>
    """, unsafe_allow_html=True)


def render_analyze_page(model, device) -> None:
    """Render the main MRI analysis page."""
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🔬 MRI Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Upload brain MRI scans for tumor detection and classification</div>", unsafe_allow_html=True)

    if model is None:
        st.markdown("""
        <div class="alert-danger">
        🚫 <b>Model not loaded.</b> Please ensure <code>saved_models/best_model.pth</code>
        exists. Train the model using <code>python train.py</code> first.
        </div>
        """, unsafe_allow_html=True)
        return

    # Analysis mode selector
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Image", "Multiple Images"],
        horizontal=True,
    )

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    if analysis_mode == "Single Image":
        render_single_analysis(model, device)
    else:
        render_batch_analysis(model, device)


def render_single_analysis(model, device) -> None:
    """Render single image analysis workflow."""
    # Upload section
    st.markdown("<div class='section-header'>📁 Upload MRI Scan</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload an MRI image",
        type=ALLOWED_TYPES,
        help="Supported formats: JPG, PNG, BMP, TIFF. Max size: 10MB",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="alert-info">
        💡 <b>How to use:</b> Upload a brain MRI image above. The AI will analyze it and
        provide tumor classification with confidence scores and Grad-CAM explainability heatmaps.
        </div>
        """, unsafe_allow_html=True)
        return

    # Validate image
    image = validate_image(uploaded_file)
    if image is None:
        return

    # Display uploaded image
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown("<div class='section-header'>Uploaded Image</div>", unsafe_allow_html=True)
        st.image(image, caption=f"📷 {uploaded_file.name}", use_column_width=True)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px | {len(uploaded_file.getvalue())/1024:.1f} KB")

    with col_info:
        st.markdown("<div class='section-header'>Analysis Settings</div>", unsafe_allow_html=True)

        cam_method = st.selectbox(
            "Explainability Method",
            ["gradcam", "gradcam++", "eigencam"],
            help="Grad-CAM variant for generating activation heatmaps",
        )

        show_all_panels = st.checkbox("Show full explainability panel", value=True)

        analyze_btn = st.button("🔬 Analyze MRI", type="primary", use_container_width=True)

    if not analyze_btn:
        return

    # ── Run analysis ──────────────────────────────────────────────────────
    with st.spinner("🧠 Analyzing MRI scan..."):
        try:
            # Save uploaded image temporarily
            temp_path = os.path.join(DEMO_IMAGE_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Run prediction
            transform = get_inference_transforms()
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image_tensor)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            predicted_idx   = probs.argmax()
            predicted_class = IDX_TO_CLASS[predicted_idx]
            confidence      = float(probs[predicted_idx])
            risk_level, risk_desc = get_risk_level(predicted_class)

            # Generate Grad-CAM
            grad_cam = BrainTumorGradCAM(model, device, method=cam_method)
            grayscale_cam, cam_image_arr, original_img = grad_cam.generate_heatmap(
                image_tensor[0],
                target_class=int(predicted_idx),
            )

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return

    # ── Display results ───────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🎯 Detection Results</div>", unsafe_allow_html=True)

    pred_color = get_class_color(predicted_class)
    pred_emoji = get_class_emoji(predicted_class)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">{pred_emoji}</div>
            <div style="font-size:0.8rem;color:#94a3b8;margin:4px 0;">Prediction</div>
            <div style="font-size:1.1rem;font-weight:700;color:{pred_color};">
                {predicted_class.replace('_',' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">📊</div>
            <div style="font-size:0.8rem;color:#94a3b8;margin:4px 0;">Confidence</div>
            <div style="font-size:1.1rem;font-weight:700;color:white;">{confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">⚠️</div>
            <div style="font-size:0.8rem;color:#94a3b8;margin:4px 0;">Risk Level</div>
            <div style="font-size:1.1rem;font-weight:700;color:{pred_color};">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">🔍</div>
            <div style="font-size:0.8rem;color:#94a3b8;margin:4px 0;">Method</div>
            <div style="font-size:1.1rem;font-weight:700;color:white;">{cam_method.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk description
    st.markdown(f"""
    <div class="alert-{'danger' if risk_level=='HIGH RISK' else 'warning' if risk_level=='MODERATE' else 'success'}">
        <b>{pred_emoji} {predicted_class.replace('_',' ').title()}</b> — {risk_desc}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Results columns: chart + CAM
    col_chart, col_cam = st.columns([1, 2])

    with col_chart:
        st.markdown("<div class='section-header'>📈 Class Probabilities</div>", unsafe_allow_html=True)
        prob_fig = create_probability_chart(probs, predicted_class)
        st.pyplot(prob_fig, clear_figure=True)
        plt.close(prob_fig)

        # Detailed probability table
        st.markdown("**Detailed Scores:**")
        for cls, prob in zip(CLASS_NAMES, probs):
            color = get_class_color(cls)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:4px 8px;margin:2px 0;background:#0f2340;border-radius:6px;">
                <span style="color:white;font-size:0.85rem;">{cls.replace('_',' ').title()}</span>
                <span style="color:{color};font-weight:700;font-size:0.9rem;">{prob*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

    with col_cam:
        st.markdown("<div class='section-header'>🗺️ Explainability — Grad-CAM</div>", unsafe_allow_html=True)
        if show_all_panels:
            cam_fig = create_gradcam_figure(
                original_img, grayscale_cam, cam_image_arr,
                predicted_class, confidence
            )
            st.pyplot(cam_fig, clear_figure=True)
            plt.close(cam_fig)
        else:
            st.image(cam_image_arr, caption="Grad-CAM Region Highlight", use_column_width=True)

        st.markdown("""
        <div style="font-size:0.8rem;color:#64748b;margin-top:8px;">
        🔥 <b>Red/hot regions</b> = areas most influential in the model's decision<br>
        🔵 <b>Blue/cool regions</b> = less important background areas
        </div>
        """, unsafe_allow_html=True)

    # ── Download Report ───────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📥 Download Report</div>", unsafe_allow_html=True)

    result_dict = {
        "predicted_class": predicted_class,
        "predicted_idx":   int(predicted_idx),
        "confidence":      confidence,
        "probabilities":   {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
    }

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        # Download CAM overlay as PNG
        cam_pil = Image.fromarray(cam_image_arr)
        cam_bytes = pil_to_bytes(cam_pil)
        st.download_button(
            "⬇️ Download Heatmap (PNG)",
            data=cam_bytes,
            file_name=f"gradcam_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True,
        )

    with col_dl2:
        # Download probability chart
        prob_fig2 = create_probability_chart(probs, predicted_class)
        chart_bytes = fig_to_bytes(prob_fig2)
        plt.close(prob_fig2)
        st.download_button(
            "⬇️ Download Chart (PNG)",
            data=chart_bytes,
            file_name=f"probabilities_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True,
        )

    with col_dl3:
        # Download PDF report
        try:
            pdf_bytes = generate_pdf_report(
                image, result_dict, cam_image_arr, original_img, grayscale_cam
            )
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"neuroscan_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ This AI prediction is for research purposes only and should NOT be used for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


def render_batch_analysis(model, device) -> None:
    """Render batch (multiple image) analysis workflow."""
    st.markdown("<div class='section-header'>📁 Upload Multiple MRI Scans</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple MRI images",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        help="Upload multiple brain MRI images for batch analysis",
    )

    if not uploaded_files:
        st.markdown("""
        <div class="alert-info">
        💡 Upload multiple MRI images for batch analysis. Results will be displayed in a table.
        </div>
        """, unsafe_allow_html=True)
        return

    st.info(f"📋 {len(uploaded_files)} image(s) uploaded")

    if not st.button("🔬 Analyze All Images", type="primary"):
        return

    results = []
    progress_bar = st.progress(0)
    status_text  = st.empty()

    transform = get_inference_transforms()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Analyzing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        progress_bar.progress((i + 1) / len(uploaded_files))

        image = validate_image(uploaded_file)
        if image is None:
            continue

        try:
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image_tensor)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx   = probs.argmax()
            pred_class = IDX_TO_CLASS[pred_idx]
            conf       = float(probs[pred_idx])

            results.append({
                "Filename":   uploaded_file.name,
                "Prediction": pred_class.replace("_", " ").title(),
                "Confidence": f"{conf*100:.1f}%",
                "Glioma %":      f"{probs[0]*100:.1f}%",
                "Meningioma %":  f"{probs[1]*100:.1f}%",
                "No Tumor %":    f"{probs[2]*100:.1f}%",
                "Pituitary %":   f"{probs[3]*100:.1f}%",
            })
        except Exception as e:
            results.append({
                "Filename":   uploaded_file.name,
                "Prediction": f"ERROR: {e}",
                "Confidence": "—",
            })

    status_text.text("✅ Analysis complete!")
    progress_bar.empty()

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        st.markdown("<div class='section-header'>📊 Batch Results</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results (CSV)",
            data=csv_bytes,
            file_name=f"batch_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )


def render_about_page() -> None:
    """Render the About Model page."""
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>📊 About the Model</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Architecture Details</div>", unsafe_allow_html=True)
        arch_info = [
            ("Backbone",        "Vision Transformer (ViT-Base/16)"),
            ("Pretrained On",   "ImageNet-21k"),
            ("Patch Size",      "16×16 pixels"),
            ("Num Patches",     "196 (14×14 grid)"),
            ("Embed Dim",       "768"),
            ("Sequence Model",  "GRU (2 layers, hidden=256)"),
            ("Input Size",      "224×224×3"),
            ("Output Classes",  "4 (Glioma, Meningioma, No Tumor, Pituitary)"),
            ("XAI Method",      "Grad-CAM (pytorch-grad-cam)"),
        ]
        for key, val in arch_info:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 10px;
                        margin:3px 0;background:#0f2340;border-radius:6px;border-left:3px solid #4fc3f7;">
                <span style="color:#94a3b8;font-size:0.85rem;">{key}</span>
                <span style="color:white;font-weight:600;font-size:0.85rem;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>Training Configuration</div>", unsafe_allow_html=True)
        train_info = [
            ("Optimizer",      "AdamW"),
            ("Learning Rate",  "1e-4"),
            ("Scheduler",      "CosineAnnealingLR"),
            ("Loss Function",  "CrossEntropyLoss + Label Smoothing"),
            ("Batch Size",     "32"),
            ("Augmentation",   "Flip, Rotate, ColorJitter, RandomCrop"),
            ("Normalization",  "ImageNet (mean=[0.485,0.456,0.406])"),
            ("Early Stopping", "Patience=10 epochs"),
            ("Dataset",        "Brain Tumor MRI (Kaggle)"),
        ]
        for key, val in train_info:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 10px;
                        margin:3px 0;background:#0f2340;border-radius:6px;border-left:3px solid #2ecc71;">
                <span style="color:#94a3b8;font-size:0.85rem;">{key}</span>
                <span style="color:white;font-weight:600;font-size:0.85rem;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>How Grad-CAM Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#94a3b8;line-height:1.8;font-size:0.9rem;">
    <b style="color:#4fc3f7">Gradient-weighted Class Activation Mapping (Grad-CAM)</b> generates visual explanations
    for CNN/ViT decisions by:
    <br><br>
    1. <b style="color:white">Forward pass</b> — Feed MRI image through the model to get a prediction<br>
    2. <b style="color:white">Backpropagate</b> — Compute gradients of the predicted class score w.r.t. the target layer<br>
    3. <b style="color:white">Weight maps</b> — Average gradients across channels to get importance weights<br>
    4. <b style="color:white">Heatmap</b> — Weighted sum of feature maps gives a spatial importance map<br>
    5. <b style="color:white">Overlay</b> — Upsample heatmap to image size and blend with the original MRI<br>
    <br>
    Red/hot regions indicate areas the model focused on to make its classification decision.
    </div>
    """, unsafe_allow_html=True)


def render_help_page() -> None:
    """Render the Help/FAQ page."""
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>❓ Help & FAQ</div>", unsafe_allow_html=True)

    faqs = [
        ("How do I use NeuroScan AI?",
         "Navigate to '🔬 Analyze MRI' in the sidebar, upload a brain MRI image (JPG/PNG), and click 'Analyze MRI'. Results with confidence scores and Grad-CAM heatmaps will appear instantly."),
        ("What image formats are supported?",
         "JPG, JPEG, PNG, BMP, and TIFF. Maximum file size is 10MB. Images should be brain MRI scans."),
        ("What does confidence score mean?",
         "Confidence is the model's probability for the predicted class (0–100%). Higher means more certainty. Values above 80% indicate high confidence."),
        ("How accurate is the model?",
         "The ViT-GRU model achieves ~95%+ accuracy on the Brain Tumor MRI dataset. However, AI predictions should never replace professional medical diagnosis."),
        ("What is Grad-CAM?",
         "Gradient-weighted Class Activation Mapping highlights which regions of the MRI scan influenced the model's prediction. Red areas = high importance for the decision."),
        ("The model is not loaded — what do I do?",
         "Ensure you've trained the model and saved it to 'saved_models/best_model.pth'. Run: python train.py --data_dir dataset/Training"),
        ("Can I use this for medical diagnosis?",
         "NO. This tool is for research and educational purposes only. Always consult a qualified radiologist or neurosurgeon for medical decisions."),
        ("How to deploy on Streamlit Cloud?",
         "1. Push code to GitHub  2. Go to share.streamlit.io  3. Connect your repo  4. Set main file as app.py  5. Add requirements.txt"),
    ]

    for question, answer in faqs:
        with st.expander(f"❓ {question}"):
            st.markdown(f"<div style='color:#94a3b8;font-size:0.9rem;line-height:1.7;'>{answer}</div>",
                        unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Quick Start Commands</div>", unsafe_allow_html=True)
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (run in Colab/terminal)
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
unzip brain-tumor-classification-mri.zip -d dataset/

# 3. Train the model
python train.py --data_dir dataset/Training --epochs 30 --batch_size 32

# 4. Run Streamlit app
streamlit run app.py

# 5. Open browser
# http://localhost:8501
    """, language="bash")


# =============================================================================
# MAIN APP ENTRY POINT
# =============================================================================

def main() -> None:
    """Main Streamlit application entry point."""
    # Inject CSS
    inject_custom_css()

    # Detect device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (cached)
    with st.spinner("Loading AI model..."):
        model, device = load_cached_model(CHECKPOINT_PATH, device_str)

    model_loaded = model is not None

    # Render sidebar and get selected page
    selected = render_sidebar(model_loaded)

    # Route to pages
    if "Home" in selected:
        render_home_page()
    elif "Analyze" in selected:
        render_analyze_page(model, device)
    elif "About" in selected:
        render_about_page()
    elif "Help" in selected:
        render_help_page()


if __name__ == "__main__":
    main()
