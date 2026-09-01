# =============================================================================
# dataset.py
# Brain Tumor Detection - Dataset Loading, Preprocessing & Augmentation
# =============================================================================
# This file handles all data pipeline operations:
#   1. Folder-based dataset loading (Kaggle brain tumor MRI dataset)
#   2. Image preprocessing (resize, normalize)
#   3. Data augmentation for training robustness
#   4. Train / Validation / Test splitting
#   5. PyTorch DataLoader creation
# =============================================================================

import os                                    # file path operations
import random                                # random seed control
from pathlib import Path                     # modern path handling
from typing import Dict, List, Tuple, Optional

import numpy as np                           # numerical operations
import pandas as pd                          # data analysis / CSV export
import cv2                                   # image loading & preprocessing
from PIL import Image                        # PIL image support
import matplotlib.pyplot as plt              # visualization
from sklearn.model_selection import train_test_split  # data splitting

import torch                                 # PyTorch core
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms  # image transformations

# =============================================================================
# CONSTANTS
# =============================================================================

# Class names must match your dataset folder names exactly
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Map class name → integer label
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Map integer label → class name (for display)
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Image dimensions expected by ViT-Base/16
IMAGE_SIZE = 224

# ImageNet normalization statistics (used since ViT is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
# DATASET CLASS
# =============================================================================

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI images.

    Expects folder structure:
        data_dir/
            glioma/        ← MRI images of glioma tumors
            meningioma/    ← MRI images of meningioma tumors
            no_tumor/      ← MRI images with no tumor
            pituitary/     ← MRI images of pituitary tumors

    Args:
        image_paths: List of file paths to MRI images
        labels:      Corresponding integer labels (0-3)
        transform:   torchvision transforms to apply
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths    # list of image file paths
        self.labels = labels              # list of integer class labels
        self.transform = transform        # augmentation/preprocessing pipeline

        # Verify lengths match
        assert len(self.image_paths) == len(self.labels), \
            "Mismatch: number of images and labels must be equal"

    def __len__(self) -> int:
        """Returns total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single (image_tensor, label) pair.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            image:  Preprocessed image tensor (3, 224, 224)
            label:  Integer class label (0–3)
        """
        # Load image path and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image using PIL (converts to RGB automatically)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: create blank image if file is corrupted
            print(f"[Warning] Failed to load {img_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        # Apply transformations (resize, normalize, augmentations)
        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# TRANSFORMS / AUGMENTATION
# =============================================================================

def get_train_transforms() -> transforms.Compose:
    """
    Training augmentation pipeline.
    Heavy augmentation helps prevent overfitting on medical images.

    Returns:
        Composed transform pipeline for training data
    """
    return transforms.Compose([
        # Resize to ViT input size
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Random horizontal flip (50% probability)
        transforms.RandomHorizontalFlip(p=0.5),

        # Random vertical flip (20% probability)
        transforms.RandomVerticalFlip(p=0.2),

        # Random rotation ±15 degrees
        transforms.RandomRotation(degrees=15),

        # Random brightness, contrast, saturation jitter
        transforms.ColorJitter(
            brightness=0.2,    # ±20% brightness
            contrast=0.2,      # ±20% contrast
            saturation=0.1,    # ±10% saturation
        ),

        # Random crop after padding (preserves full image content)
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.85, 1.0),    # keep 85–100% of original
            ratio=(0.9, 1.1),     # slight aspect ratio variation
        ),

        # Convert PIL Image → Tensor (scales pixels to [0,1])
        transforms.ToTensor(),

        # Normalize with ImageNet stats (required for pretrained ViT)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms() -> transforms.Compose:
    """
    Validation / Test transform pipeline.
    No augmentation — only resize and normalize for fair evaluation.

    Returns:
        Composed transform pipeline for val/test data
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),    # resize to ViT input
        transforms.ToTensor(),                           # PIL → Tensor
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # normalize
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Inference transforms for single-image prediction (Streamlit app).
    Identical to val/test — no augmentation.
    """
    return get_val_test_transforms()


# =============================================================================
# DATASET LOADING UTILITIES
# =============================================================================

def scan_dataset_directory(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scan a dataset directory and collect all image paths with labels.

    Expected folder structure:
        data_dir/
            class_name_1/  image1.jpg  image2.jpg ...
            class_name_2/  image1.jpg  image2.jpg ...
            ...

    Args:
        data_dir: Root directory containing class subfolders

    Returns:
        image_paths: List of absolute image file paths
        labels:      Corresponding integer class labels
    """
    data_dir = Path(data_dir)

    # Verify the directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_paths = []
    labels = []
    class_counts = {}

    # Supported image file extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Iterate over each class folder
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name

        # Skip if class folder doesn't exist
        if not class_dir.exists():
            print(f"[Warning] Class folder not found: {class_dir}")
            continue

        # Collect all image files in this class folder
        class_images = [
            str(f) for f in class_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ]

        # Sort for reproducibility
        class_images.sort()

        # Get integer label for this class
        label = CLASS_TO_IDX[class_name]

        # Add to master lists
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))
        class_counts[class_name] = len(class_images)

    # Print dataset statistics
    print("\n[Dataset] Scan complete:")
    total = 0
    for cls, count in class_counts.items():
        print(f"  {cls:<15}: {count:>5} images")
        total += count
    print(f"  {'TOTAL':<15}: {total:>5} images\n")

    return image_paths, labels


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List, List, List, List, List, List]:
    """
    Split dataset into train / validation / test sets.
    Stratified split preserves class distribution in each split.

    Args:
        image_paths:  All image file paths
        labels:       Corresponding labels
        train_ratio:  Fraction for training   (default 70%)
        val_ratio:    Fraction for validation (default 15%)
        test_ratio:   Fraction for testing    (default 15%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, val_paths, test_paths,
                  train_labels, val_labels, test_labels)
    """
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: separate test set
    test_size = test_ratio
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,    # maintain class distribution
    )

    # Second split: separate validation from remaining
    val_size = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    # Print split statistics
    print(f"[Dataset] Split summary:")
    print(f"  Train      : {len(train_paths):>5} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation : {len(val_paths):>5} samples ({val_ratio*100:.0f}%)")
    print(f"  Test       : {len(test_paths):>5} samples ({test_ratio*100:.0f}%)\n")

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Ensures each class is sampled equally during training.

    Args:
        labels: List of integer class labels for training set

    Returns:
        WeightedRandomSampler for use in DataLoader
    """
    # Count samples per class
    class_counts = np.bincount(labels)

    # Weight = 1 / class_frequency (rare classes get higher weight)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # normalize

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.FloatTensor(sample_weights)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print(f"[Dataset] Class weights: { {IDX_TO_CLASS[i]: f'{w:.4f}' for i, w in enumerate(class_weights)} }")

    return sampler


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Main function: scan data directory and create train/val/test DataLoaders.

    Args:
        data_dir:             Root dataset directory path
        batch_size:           Number of samples per batch
        num_workers:          Parallel data loading workers
        use_weighted_sampler: Balance classes via weighted sampling
        pin_memory:           Pin memory for faster GPU transfer

    Returns:
        train_loader:  DataLoader for training
        val_loader:    DataLoader for validation
        test_loader:   DataLoader for testing
        dataset_info:  Dictionary with dataset statistics
    """
    # ── Step 1: Scan directory for all images and labels ──────────────────
    image_paths, labels = scan_dataset_directory(data_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Check folder structure.")

    # ── Step 2: Split into train / val / test ─────────────────────────────
    train_paths, val_paths, test_paths, \
    train_labels, val_labels, test_labels = split_dataset(image_paths, labels)

    # ── Step 3: Create transform pipelines ────────────────────────────────
    train_transforms = get_train_transforms()
    val_test_transforms = get_val_test_transforms()

    # ── Step 4: Create Dataset objects ────────────────────────────────────
    train_dataset = BrainTumorDataset(train_paths, train_labels, train_transforms)
    val_dataset   = BrainTumorDataset(val_paths,   val_labels,   val_test_transforms)
    test_dataset  = BrainTumorDataset(test_paths,  test_labels,  val_test_transforms)

    # ── Step 5: Optional weighted sampler for class balance ───────────────
    sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_labels)
        shuffle_train = False  # cannot use both shuffle=True and sampler

    # ── Step 6: Create DataLoaders ────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,    # shuffle if no sampler
        sampler=sampler,           # weighted sampler (or None)
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,            # drop incomplete last batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,             # no shuffling for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ── Step 7: Collect dataset info ──────────────────────────────────────
    dataset_info = {
        "num_classes":   len(CLASS_NAMES),
        "class_names":   CLASS_NAMES,
        "class_to_idx":  CLASS_TO_IDX,
        "idx_to_class":  IDX_TO_CLASS,
        "train_size":    len(train_dataset),
        "val_size":      len(val_dataset),
        "test_size":     len(test_dataset),
        "total_size":    len(image_paths),
        "batch_size":    batch_size,
        "image_size":    IMAGE_SIZE,
    }

    print(f"[Dataset] DataLoaders created successfully.")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}\n")

    return train_loader, val_loader, test_loader, dataset_info


# =============================================================================
# VISUALIZATION UTILITY
# =============================================================================

def visualize_samples(
    dataset: BrainTumorDataset,
    num_samples: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """
    Display a grid of sample images with their class labels.

    Args:
        dataset:     BrainTumorDataset instance
        num_samples: Number of samples to display (default: 8)
        save_path:   Optional path to save the visualization
    """
    # Select random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Create subplot grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    # ImageNet denormalization for visualization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for i, idx in enumerate(indices):
        image_tensor, label = dataset[idx]

        # Denormalize: reverse ImageNet normalization
        image_display = image_tensor * std + mean
        image_display = image_display.permute(1, 2, 0).numpy()  # CHW → HWC
        image_display = np.clip(image_display, 0, 1)              # clip to [0,1]

        axes[i].imshow(image_display)
        axes[i].set_title(
            f"Class: {IDX_TO_CLASS[label]}\nLabel: {label}",
            fontsize=10
        )
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Brain Tumor MRI Sample Images", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved to {save_path}")

    plt.show()


def save_dataset_stats(
    dataset_info: Dict,
    labels: List[int],
    save_dir: str = "outputs/",
) -> None:
    """
    Save dataset statistics as CSV and distribution plot.

    Args:
        dataset_info: Dictionary returned by create_dataloaders
        labels:       All integer labels (full dataset)
        save_dir:     Directory to save outputs
    """
    os.makedirs(save_dir, exist_ok=True)

    # Count per-class samples
    class_counts = np.bincount(labels)

    # Create DataFrame
    df = pd.DataFrame({
        "Class":       CLASS_NAMES,
        "Label Index": list(range(len(CLASS_NAMES))),
        "Count":       class_counts,
        "Percentage":  (class_counts / class_counts.sum() * 100).round(2),
    })

    # Save CSV
    csv_path = os.path.join(save_dir, "dataset_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Stats] Dataset stats saved: {csv_path}")

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(CLASS_NAMES, class_counts, color=colors, edgecolor="black", linewidth=0.7)

    # Add count labels above bars
    for bar, count in zip(bars, class_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Brain Tumor Dataset — Class Distribution", fontsize=14, pad=15)
    ax.set_xlabel("Tumor Class", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_ylim(0, max(class_counts) * 1.15)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[Stats] Distribution plot saved: {plot_path}")
    plt.show()


# =============================================================================
# KAGGLE DOWNLOAD HELPER
# =============================================================================

def download_kaggle_dataset(kaggle_dataset: str = "sartajbhuvaji/brain-tumor-classification-mri") -> None:
    """
    Instructions and code to download the brain tumor dataset from Kaggle.
    Run this in Google Colab with your Kaggle API credentials.

    Args:
        kaggle_dataset: Kaggle dataset identifier (user/dataset-name)
    """
    instructions = f"""
    =========================================================
    HOW TO DOWNLOAD THE BRAIN TUMOR MRI DATASET FROM KAGGLE
    =========================================================

    STEP 1: Get your Kaggle API credentials
      - Go to https://www.kaggle.com → Account → Create New API Token
      - This downloads kaggle.json

    STEP 2: Upload kaggle.json to Colab
      from google.colab import files
      files.upload()   # select your kaggle.json file

    STEP 3: Set up Kaggle credentials
      !mkdir -p ~/.kaggle
      !cp kaggle.json ~/.kaggle/
      !chmod 600 ~/.kaggle/kaggle.json

    STEP 4: Download the dataset
      !pip install kaggle -q
      !kaggle datasets download -d {kaggle_dataset}
      !unzip brain-tumor-classification-mri.zip -d dataset/

    STEP 5: Verify folder structure
      !ls dataset/Training/

    Expected output:
      glioma/   meningioma/   no_tumor/   pituitary/

    STEP 6: Use data_dir in this project
      data_dir = "dataset/Training"
    =========================================================
    """
    print(instructions)


# =============================================================================
# MAIN: Test the dataset pipeline
# =============================================================================
if __name__ == "__main__":
    # Show download instructions
    download_kaggle_dataset()

    # Example: test with a sample directory
    # Uncomment and modify the path after downloading your dataset
    # data_dir = "dataset/Training"
    # train_loader, val_loader, test_loader, info = create_dataloaders(
    #     data_dir=data_dir, batch_size=32
    # )
    # print("Dataset pipeline test passed ✓")

    print("dataset.py loaded successfully ✓")
