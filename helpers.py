# =============================================================================
# utils/helpers.py
# Brain Tumor Detection — Shared Utility Functions
# =============================================================================

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Seed] All random seeds set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available compute device.

    Args:
        prefer_gpu: Use GPU if available

    Returns:
        torch.device instance
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print(f"[Device] Using CPU")

    return device


def create_project_directories(base_dir: str = ".") -> Dict[str, str]:
    """
    Create all required project directories.

    Args:
        base_dir: Base project directory

    Returns:
        Dictionary mapping directory names to their paths
    """
    dirs = {
        "dataset":       os.path.join(base_dir, "dataset"),
        "saved_models":  os.path.join(base_dir, "saved_models"),
        "uploads":       os.path.join(base_dir, "uploads"),
        "outputs":       os.path.join(base_dir, "outputs"),
        "utils":         os.path.join(base_dir, "utils"),
        "notebooks":     os.path.join(base_dir, "notebooks"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[Setup] Directory ready: {path}")

    return dirs


def save_json(data: Dict, path: str) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[JSON] Saved → {path}")


def load_json(path: str) -> Dict:
    """Load JSON file to dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Dict with total, trainable, frozen parameter counts
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"[Params] Total:     {total:>12,}")
    print(f"[Params] Trainable: {trainable:>12,}")
    print(f"[Params] Frozen:    {frozen:>12,}")

    return {"total": total, "trainable": trainable, "frozen": frozen}


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Image tensor (3, H, W) with ImageNet normalization

    Returns:
        NumPy array (H, W, 3) in range [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = tensor.cpu() * std + mean
    tensor = tensor.permute(1, 2, 0).numpy()
    return np.clip(tensor, 0, 1)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


if __name__ == "__main__":
    # Quick self-test
    set_seed(42)
    device = get_device()
    dirs   = create_project_directories()
    print("\nutils/helpers.py loaded successfully ✓")
