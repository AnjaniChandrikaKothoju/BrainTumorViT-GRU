# =============================================================================
# explainability.py
# Brain Tumor Detection - Grad-CAM Explainable AI Visualizations
# =============================================================================
# This file implements Explainable AI using pytorch-grad-cam:
#   1. Extracts Grad-CAM heatmaps from the ViT backbone
#   2. Overlays heatmaps on original MRI images
#   3. Highlights tumor-relevant regions
#   4. Saves explainability visualizations
#
# Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradient
# of the predicted class score with respect to the final convolutional/attention
# feature maps. Regions with high gradient magnitude are most important for
# the model's decision.
# =============================================================================

import os
import logging
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# pytorch-grad-cam library
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,        # overlay CAM on image
    preprocess_image,          # standard preprocessing helper
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # target class

# Local modules
from dataset import (
    get_inference_transforms,
    IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, IDX_TO_CLASS,
    IMAGE_SIZE,
)

logger = logging.getLogger("BrainTumorTrainer")


# =============================================================================
# ViT RESHAPE TRANSFORM
# =============================================================================

class ViTReshapeTransform:
    """
    Reshape transform required for Vision Transformer Grad-CAM.

    ViT produces token sequences of shape (B, num_tokens, embed_dim).
    Grad-CAM expects spatial feature maps of shape (B, C, H, W).

    This transform:
      1. Removes the [CLS] token (index 0)
      2. Reshapes the remaining 196 patch tokens into a 14×14 spatial grid
      3. Transposes from (B, H*W, C) → (B, C, H, W)

    The resulting (B, embed_dim, 14, 14) tensor can be processed by Grad-CAM
    just like a CNN feature map.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: The HybridViTGRU model (for getting embed_dim)
        """
        # Number of patches along each dimension: 224 / 16 = 14
        self.patches_h = 14
        self.patches_w = 14

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape ViT token sequence to spatial feature map.

        Args:
            x: ViT output tokens (B, num_tokens, embed_dim)
               typically (B, 197, 768)

        Returns:
            Spatial feature map (B, embed_dim, 14, 14)
        """
        # Remove [CLS] token at position 0
        # x[:, 1:, :] → (B, 196, 768)
        x = x[:, 1:, :]

        batch_size, num_patches, embed_dim = x.shape

        # Reshape: (B, 196, 768) → (B, 14, 14, 768)
        x = x.reshape(batch_size, self.patches_h, self.patches_w, embed_dim)

        # Transpose to channel-first: (B, 14, 14, 768) → (B, 768, 14, 14)
        x = x.permute(0, 3, 1, 2)

        return x


# =============================================================================
# GRAD-CAM WRAPPER
# =============================================================================

class BrainTumorGradCAM:
    """
    Grad-CAM explainability for the Hybrid ViT-GRU model.

    Wraps pytorch-grad-cam to work with the ViT backbone,
    producing heatmaps that highlight tumor-relevant regions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        method: str = "gradcam",
    ):
        """
        Initialize Grad-CAM with the target layer in the ViT backbone.

        Args:
            model:  HybridViTGRU model instance
            device: Compute device
            method: CAM method — 'gradcam', 'gradcam++', or 'eigencam'
        """
        self.model = model
        self.device = device
        self.model.eval()   # ensure eval mode

        # ── Target Layer ──────────────────────────────────────────────────
        # We target the last transformer block's attention norm in ViT.
        # This is the deepest layer before the final patch embeddings,
        # capturing the highest-level spatial features.
        #
        # For vit_base_patch16_224 (timm):
        #   model.vit.blocks[-1].norm1  ← last transformer block norm
        #   or model.vit.blocks[-1]     ← last full transformer block
        self.target_layer = [model.vit.blocks[-1].norm1]

        # ── Reshape transform for ViT ──────────────────────────────────────
        self.reshape_transform = ViTReshapeTransform(model)

        # ── Select CAM method ─────────────────────────────────────────────
        cam_methods = {
            "gradcam":   GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "eigencam":  EigenCAM,
        }

        if method not in cam_methods:
            raise ValueError(f"Unknown CAM method: {method}. Choose from {list(cam_methods.keys())}")

        # Initialize pytorch-grad-cam
        self.cam = cam_methods[method](
            model=model,
            target_layers=self.target_layer,
            reshape_transform=self.reshape_transform,
        )

        print(f"[GradCAM] Initialized with method='{method}' on layer: {type(self.target_layer[0]).__name__}")

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224) or (3, 224, 224)
            target_class: Class index to generate CAM for.
                          If None, uses the predicted class (argmax).

        Returns:
            grayscale_cam: Raw heatmap array (224, 224) in [0, 1]
            cam_image:     Heatmap overlaid on original image (224, 224, 3)
        """
        # Ensure batch dimension
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 224, 224)

        image_tensor = image_tensor.to(self.device)

        # Set up target class
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = None  # defaults to predicted class

        # Generate CAM
        # grayscale_cam: (1, 224, 224) numpy array
        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=targets,
        )
        grayscale_cam = grayscale_cam[0, :]  # remove batch dim → (224, 224)

        # ── Prepare original image for overlay ───────────────────────────
        # Denormalize: reverse ImageNet normalization for display
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_display = image_tensor[0].cpu() * std + mean  # (3, 224, 224)
        img_display = img_display.permute(1, 2, 0).numpy()   # (224, 224, 3)
        img_display = np.clip(img_display, 0, 1).astype(np.float32)

        # ── Overlay heatmap on image ──────────────────────────────────────
        # show_cam_on_image blends the heatmap with the original image
        cam_image = show_cam_on_image(
            img_display,      # original image in [0, 1]
            grayscale_cam,    # heatmap in [0, 1]
            use_rgb=True,     # output as RGB
            colormap=cv2.COLORMAP_JET,   # red=hot, blue=cold
            image_weight=0.5, # blend weight for original image
        )

        return grayscale_cam, cam_image, img_display

    def generate_batch_heatmaps(
        self,
        image_tensors: torch.Tensor,
        target_classes: Optional[List[int]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Args:
            image_tensors: Batch tensor (B, 3, 224, 224)
            target_classes: List of target class indices (or None for predicted)

        Returns:
            List of (grayscale_cam, cam_image, img_display) tuples
        """
        results = []
        for i in range(image_tensors.shape[0]):
            target = target_classes[i] if target_classes else None
            result = self.generate_heatmap(image_tensors[i], target_class=target)
            results.append(result)
        return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_explainability_figure(
    original_image: np.ndarray,
    grayscale_cam: np.ndarray,
    cam_image: np.ndarray,
    predicted_class: str,
    confidence: float,
    probabilities: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """
    Create a comprehensive explainability figure showing:
      Panel 1: Original MRI image
      Panel 2: Grad-CAM heatmap (standalone)
      Panel 3: Heatmap overlaid on MRI
      Panel 4: Prediction probability bar chart

    Args:
        original_image:   Original MRI image array (224, 224, 3) in [0, 1]
        grayscale_cam:    Raw CAM heatmap (224, 224) in [0, 1]
        cam_image:        CAM overlaid on image (224, 224, 3)
        predicted_class:  Predicted tumor class name
        confidence:       Prediction confidence score (0–1)
        probabilities:    All class probabilities (4,)
        save_path:        Optional path to save the figure
        figsize:          Figure size (width, height) in inches

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)

    # Color coding for classes
    class_colors = {
        "glioma":     "#e74c3c",  # red
        "meningioma": "#3498db",  # blue
        "no_tumor":   "#2ecc71",  # green
        "pituitary":  "#f39c12",  # orange
    }
    pred_color = class_colors.get(predicted_class, "#9b59b6")

    # ── Panel 1: Original MRI ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_image)
    ax1.set_title("Original MRI", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")

    # ── Panel 2: Grad-CAM Heatmap ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    # Custom red-hot colormap for medical imaging
    medical_cmap = plt.cm.hot
    heatmap_display = ax2.imshow(grayscale_cam, cmap=medical_cmap, vmin=0, vmax=1)
    ax2.set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")
    plt.colorbar(heatmap_display, ax=ax2, fraction=0.046, pad=0.04)

    # ── Panel 3: Overlay ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(cam_image)
    ax3.set_title(
        f"Region Highlight\nPredicted: {predicted_class.replace('_', ' ').title()}\n"
        f"Confidence: {confidence*100:.1f}%",
        fontsize=10, fontweight="bold", pad=8, color=pred_color
    )
    ax3.axis("off")

    # Add a colored border to the prediction panel
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(pred_color)
        spine.set_linewidth(3)

    # ── Panel 4: Probability Chart ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    colors = [class_colors.get(c, "#9b59b6") for c in CLASS_NAMES]
    bars = ax4.barh(
        [c.replace("_", " ").title() for c in CLASS_NAMES],
        probabilities * 100,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )

    # Highlight predicted class bar
    pred_idx = CLASS_NAMES.index(predicted_class) if predicted_class in CLASS_NAMES else 0
    bars[pred_idx].set_linewidth(2)
    bars[pred_idx].set_edgecolor("black")

    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        ax4.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob*100:.1f}%",
            va="center", ha="left", fontsize=9, fontweight="bold"
        )

    ax4.set_xlim(0, 115)
    ax4.set_xlabel("Confidence (%)", fontsize=9)
    ax4.set_title("Class Probabilities", fontsize=11, fontweight="bold", pad=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.grid(axis="x", alpha=0.3)

    # ── Overall title ─────────────────────────────────────────────────────
    fig.suptitle(
        f"Brain Tumor Detection — Explainable AI Analysis",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[Explainability] Visualization saved → {save_path}")

    return fig


def visualize_multiple_samples(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    device: torch.device,
    save_dir: str = "outputs/",
    num_samples: int = 4,
    cam_method: str = "gradcam",
) -> None:
    """
    Generate Grad-CAM visualizations for multiple samples in a grid.

    Args:
        model:       HybridViTGRU model
        images:      Batch of image tensors (B, 3, 224, 224)
        labels:      Ground truth labels (B,)
        probabilities: Model output probabilities (B, 4)
        device:      Compute device
        save_dir:    Directory to save visualizations
        num_samples: Number of samples to visualize
        cam_method:  Grad-CAM variant to use
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Grad-CAM
    grad_cam = BrainTumorGradCAM(model, device, method=cam_method)

    # Select samples
    num_samples = min(num_samples, images.shape[0])

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        image_tensor = images[i]
        true_label   = labels[i].item()
        probs        = probabilities[i].numpy()
        pred_idx     = probs.argmax()

        # Generate CAM
        grayscale_cam, cam_image, img_display = grad_cam.generate_heatmap(image_tensor)

        # Row title info
        true_name = IDX_TO_CLASS[true_label]
        pred_name = IDX_TO_CLASS[pred_idx]
        correct   = "✓" if true_label == pred_idx else "✗"

        # Original image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"MRI | True: {true_name}", fontsize=9)
        axes[i, 0].axis("off")

        # Heatmap
        axes[i, 1].imshow(grayscale_cam, cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[i, 1].axis("off")

        # Overlay
        color = "green" if true_label == pred_idx else "red"
        axes[i, 2].imshow(cam_image)
        axes[i, 2].set_title(
            f"Pred: {pred_name} ({probs[pred_idx]*100:.1f}%) {correct}",
            fontsize=9, color=color
        )
        axes[i, 2].axis("off")

    plt.suptitle("Grad-CAM Explainability — Multiple Samples", fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "gradcam_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Explainability] Multi-sample visualization saved → {save_path}")
    plt.close()


# =============================================================================
# SINGLE IMAGE EXPLAINABILITY (for Streamlit app)
# =============================================================================

def explain_single_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    save_path: Optional[str] = None,
    cam_method: str = "gradcam",
) -> Tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full explainability pipeline on a single MRI image.
    This is the main function called from the Streamlit app.

    Args:
        model:      Trained HybridViTGRU model
        image_path: Path to input MRI image
        device:     Compute device
        save_path:  Optional path to save the visualization
        cam_method: Grad-CAM variant

    Returns:
        predicted_class: Predicted tumor class name
        confidence:      Confidence of the top prediction
        probabilities:   All class probabilities (4,)
        grayscale_cam:   Raw heatmap (224, 224)
        cam_image:       Overlay image (224, 224, 3)
        original_img:    Original normalized image (224, 224, 3)
    """
    # ── Load and preprocess image ─────────────────────────────────────────
    transform = get_inference_transforms()

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0)  # (1, 3, 224, 224)
    image_tensor = image_tensor.to(device)

    # ── Run model inference ───────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)                           # (1, 4)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (4,)

    predicted_idx   = probs.argmax()
    predicted_class = IDX_TO_CLASS[predicted_idx]
    confidence      = float(probs[predicted_idx])

    print(f"[Explainability] Predicted: {predicted_class} ({confidence*100:.1f}%)")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"  {name:<15}: {prob*100:.2f}%")

    # ── Generate Grad-CAM ─────────────────────────────────────────────────
    grad_cam = BrainTumorGradCAM(model, device, method=cam_method)
    grayscale_cam, cam_image, original_img = grad_cam.generate_heatmap(
        image_tensor[0],
        target_class=predicted_idx,
    )

    # ── Save visualization ────────────────────────────────────────────────
    if save_path:
        create_explainability_figure(
            original_image=original_img,
            grayscale_cam=grayscale_cam,
            cam_image=cam_image,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probs,
            save_path=save_path,
        )

    return predicted_class, confidence, probs, grayscale_cam, cam_image, original_img


# =============================================================================
# MAIN: Test explainability pipeline
# =============================================================================
if __name__ == "__main__":
    from vit_gru_model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = create_model(num_classes=4, pretrained=True, device=str(device))

    # Test with a dummy image (replace with a real MRI path)
    # result = explain_single_image(
    #     model=model,
    #     image_path="path/to/mri.jpg",
    #     device=device,
    #     save_path="outputs/test_explainability.png",
    # )

    print("explainability.py loaded successfully ✓")
