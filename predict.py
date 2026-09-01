# =============================================================================
# predict.py
# Brain Tumor Detection - Inference / Prediction Pipeline
# =============================================================================
# This file provides:
#   1. Single-image prediction function
#   2. Batch prediction function
#   3. Model loading utilities
#   4. Command-line inference tool
# =============================================================================

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Local modules
from vit_gru_model import create_model, HybridViTGRU
from dataset import (
    get_inference_transforms,
    CLASS_NAMES, IDX_TO_CLASS,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    num_classes: int = 4,
) -> HybridViTGRU:
    """
    Load a trained HybridViTGRU model from a checkpoint file.

    Supports two checkpoint formats:
      1. Full checkpoint dict (saved by train.py): contains 'model_state_dict'
      2. Plain state dict (saved by early stopper): direct state dict

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device:          Target device. Auto-detects if None.
        num_classes:     Number of output classes (default: 4)

    Returns:
        Loaded model in eval() mode on the specified device
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize model architecture (no pretrained weights — we load our own)
    model = create_model(num_classes=num_classes, pretrained=False, device=str(device))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint with metadata
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[Predict] Loaded full checkpoint from {checkpoint_path}")
        if "best_val_acc" in checkpoint:
            logger.info(f"[Predict] Best val accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        # Plain state dict
        model.load_state_dict(checkpoint)
        logger.info(f"[Predict] Loaded state dict from {checkpoint_path}")

    # Switch to eval mode: disables dropout, uses running batch norm stats
    model.eval()
    logger.info(f"[Predict] Model ready on {device}")

    return model


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_single(
    model: HybridViTGRU,
    image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
    device: torch.device,
    return_all_probs: bool = True,
) -> Dict:
    """
    Predict tumor class for a single MRI image.

    Accepts multiple input formats:
      - File path (str or Path)
      - PIL Image
      - NumPy array (H, W, C) or (H, W)
      - PyTorch tensor (3, H, W) already preprocessed

    Args:
        model:           Trained HybridViTGRU in eval mode
        image_input:     Image in any supported format
        device:          Compute device
        return_all_probs: Include all class probabilities in result

    Returns:
        Dictionary containing:
          - predicted_class:  Predicted class name (str)
          - predicted_idx:    Predicted class index (int)
          - confidence:       Confidence of top prediction (float)
          - probabilities:    Dict of {class_name: probability}
          - top_k:            Top-k predictions with scores
    """
    transform = get_inference_transforms()

    # ── Handle different input types ──────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        # Load from file path
        image_pil = Image.open(str(image_input)).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0)  # (1, 3, 224, 224)

    elif isinstance(image_input, Image.Image):
        # PIL Image
        image_tensor = transform(image_input).unsqueeze(0)

    elif isinstance(image_input, np.ndarray):
        # NumPy array → PIL → transform
        if image_input.ndim == 2:
            # Grayscale → RGB
            image_input = np.stack([image_input] * 3, axis=-1)
        image_pil = Image.fromarray(image_input.astype(np.uint8))
        image_tensor = transform(image_pil).unsqueeze(0)

    elif isinstance(image_input, torch.Tensor):
        # Already a tensor
        if image_input.ndim == 3:
            image_tensor = image_input.unsqueeze(0)  # add batch dim
        else:
            image_tensor = image_input

    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # ── Inference ─────────────────────────────────────────────────────────
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)                            # (1, 4)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy() # (4,)

    # ── Build result ──────────────────────────────────────────────────────
    predicted_idx   = int(probs.argmax())
    predicted_class = IDX_TO_CLASS[predicted_idx]
    confidence      = float(probs[predicted_idx])

    # All class probabilities as a dict
    prob_dict = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

    # Top-k predictions sorted by confidence
    sorted_indices = probs.argsort()[::-1]
    top_k = [
        {
            "rank":       i + 1,
            "class":      IDX_TO_CLASS[idx],
            "confidence": float(probs[idx]),
            "percentage": f"{probs[idx]*100:.2f}%",
        }
        for i, idx in enumerate(sorted_indices)
    ]

    result = {
        "predicted_class": predicted_class,
        "predicted_idx":   predicted_idx,
        "confidence":      confidence,
        "confidence_pct":  f"{confidence*100:.2f}%",
        "probabilities":   prob_dict,
        "top_k":           top_k,
        "raw_probs":       probs,  # numpy array for plotting
    }

    return result


def predict_batch(
    model: HybridViTGRU,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Run prediction on a list of image files.

    Args:
        model:       Trained HybridViTGRU in eval mode
        image_paths: List of image file paths
        device:      Compute device
        batch_size:  Number of images to process at once

    Returns:
        List of prediction result dictionaries (one per image)
    """
    from torch.utils.data import DataLoader
    from dataset import BrainTumorDataset, get_inference_transforms

    transform = get_inference_transforms()

    # Create temporary dataset with dummy labels
    temp_dataset = BrainTumorDataset(
        image_paths=image_paths,
        labels=[0] * len(image_paths),  # dummy labels
        transform=transform,
    )

    temp_loader = DataLoader(
        temp_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    all_results = []
    processed = 0

    model.eval()
    with torch.no_grad():
        for images, _ in temp_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(images.shape[0]):
                idx = processed + i
                pred_idx   = probs[i].argmax()
                pred_class = IDX_TO_CLASS[pred_idx]
                conf       = float(probs[i][pred_idx])

                result = {
                    "image_path":    image_paths[idx],
                    "predicted_class": pred_class,
                    "predicted_idx": int(pred_idx),
                    "confidence":    conf,
                    "probabilities": {
                        name: float(p)
                        for name, p in zip(CLASS_NAMES, probs[i])
                    },
                }
                all_results.append(result)

            processed += images.shape[0]
            logger.info(f"[Predict] Processed {processed}/{len(image_paths)} images")

    return all_results


# =============================================================================
# DISPLAY & SAVE
# =============================================================================

def display_prediction(
    result: Dict,
    image_input: Union[str, Image.Image],
    save_path: Optional[str] = None,
) -> None:
    """
    Display prediction result with the original image and probability chart.

    Args:
        result:      Prediction dictionary from predict_single
        image_input: Original image (path or PIL)
        save_path:   Optional path to save the display figure
    """
    # Load image for display
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input

    # Class colors
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Original image ──────────────────────────────────────────────
    axes[0].imshow(image)
    pred_class = result["predicted_class"]
    conf       = result["confidence"]
    color      = colors[result["predicted_idx"]]
    axes[0].set_title(
        f"Prediction: {pred_class.replace('_',' ').title()}\n"
        f"Confidence: {conf*100:.1f}%",
        fontsize=12, color=color, fontweight="bold"
    )
    axes[0].axis("off")

    # ── Right: Probability bar chart ──────────────────────────────────────
    probs = result["raw_probs"]
    class_labels = [c.replace("_", " ").title() for c in CLASS_NAMES]
    bars = axes[1].barh(class_labels, probs * 100, color=colors, height=0.5)
    axes[1].set_xlim(0, 115)
    axes[1].set_xlabel("Confidence (%)", fontsize=10)
    axes[1].set_title("Class Probabilities", fontsize=12, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    for bar, prob in zip(bars, probs):
        axes[1].text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob*100:.1f}%",
            va="center", ha="left", fontsize=9
        )

    plt.suptitle("Brain Tumor Detection Result", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[Predict] Saved prediction figure → {save_path}")

    plt.show()
    plt.close()


def print_prediction_summary(result: Dict, image_path: Optional[str] = None) -> None:
    """Print a formatted prediction summary to the console."""
    print("\n" + "=" * 50)
    print("  BRAIN TUMOR DETECTION RESULT")
    print("=" * 50)
    if image_path:
        print(f"  Image       : {os.path.basename(image_path)}")
    print(f"  Prediction  : {result['predicted_class'].replace('_',' ').title()}")
    print(f"  Confidence  : {result['confidence_pct']}")
    print(f"\n  All Class Probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:<15}: {bar:<30} {prob*100:.2f}%")
    print("=" * 50 + "\n")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain Tumor Prediction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",        type=str, required=True,
        help="Path to MRI image or directory of images"
    )
    parser.add_argument(
        "--checkpoint",   type=str, default="saved_models/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",   type=str, default="outputs/",
        help="Directory to save prediction results"
    )
    parser.add_argument(
        "--save_json",    action="store_true",
        help="Save prediction results as JSON"
    )
    parser.add_argument(
        "--no_plot",      action="store_true",
        help="Skip saving prediction plot"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device=device)

    # Check if input is file or directory
    input_path = Path(args.image)
    os.makedirs(args.output_dir, exist_ok=True)

    if input_path.is_file():
        # Single image prediction
        result = predict_single(model, str(input_path), device)
        print_prediction_summary(result, str(input_path))

        if not args.no_plot:
            save_fig_path = os.path.join(args.output_dir, f"prediction_{input_path.stem}.png")
            display_prediction(result, str(input_path), save_path=save_fig_path)

        if args.save_json:
            # Convert numpy array to list for JSON serialization
            result_json = {k: v for k, v in result.items() if k != "raw_probs"}
            json_path = os.path.join(args.output_dir, f"prediction_{input_path.stem}.json")
            with open(json_path, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"[Predict] JSON saved → {json_path}")

    elif input_path.is_dir():
        # Batch prediction on directory
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            str(f) for f in sorted(input_path.iterdir())
            if f.suffix.lower() in valid_exts
        ]
        print(f"[Predict] Found {len(image_files)} images in {input_path}")

        results = predict_batch(model, image_files, device)

        # Print summary
        for r in results:
            print(f"  {os.path.basename(r['image_path']):<30} → {r['predicted_class']:<15} ({r['confidence']*100:.1f}%)")

        # Save JSON results
        if args.save_json:
            json_path = os.path.join(args.output_dir, "batch_predictions.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[Predict] Batch results saved → {json_path}")

    else:
        print(f"[Error] Path not found: {input_path}")
        sys.exit(1)

    print("[Predict] Done ✓")
