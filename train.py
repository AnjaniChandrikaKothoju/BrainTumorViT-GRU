# =============================================================================
# train.py
# Brain Tumor Detection - Complete Training Pipeline
# =============================================================================
# This file implements the full training loop including:
#   1. Training loop with gradient accumulation
#   2. Validation loop with metric tracking
#   3. Early stopping to prevent overfitting
#   4. Learning rate scheduling
#   5. Model checkpoint saving
#   6. Training curve visualization
# =============================================================================

import os                           # file system operations
import sys                          # system operations
import time                         # timing
import logging                      # logging
import argparse                     # command-line arguments
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")               # non-interactive backend for Colab

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns               # confusion matrix heatmap

# Local modules
from vit_gru_model import create_model, get_model_summary
from dataset import create_dataloaders, CLASS_NAMES, IDX_TO_CLASS


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: str = "outputs/") -> logging.Logger:
    """
    Configure logging to both console and file.

    Args:
        log_dir: Directory to save log file

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    # Create logger
    logger = logging.getLogger("BrainTumorTrainer")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Monitor validation loss and stop training when no improvement is seen.

    Prevents overfitting by stopping training when val_loss stops improving
    for `patience` consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 10,        # epochs to wait before stopping
        min_delta: float = 1e-4,   # minimum improvement threshold
        verbose: bool = True,       # print messages
        save_path: str = "saved_models/best_model.pth",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path

        self.counter = 0            # counts epochs without improvement
        self.best_loss = None       # best validation loss seen so far
        self.early_stop = False     # flag to stop training
        self.best_acc = 0.0         # best validation accuracy

    def __call__(self, val_loss: float, val_acc: float, model: nn.Module) -> bool:
        """
        Check whether to stop training and save best model.

        Args:
            val_loss: Current epoch validation loss
            val_acc:  Current epoch validation accuracy
            model:    Current model state

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            # First epoch: just save
            self.best_loss = val_loss
            self.best_acc = val_acc
            self._save_checkpoint(model, val_loss, val_acc)

        elif val_loss < self.best_loss - self.min_delta:
            # Improvement found: reset counter and save
            self.best_loss = val_loss
            self.best_acc = val_acc
            self._save_checkpoint(model, val_loss, val_acc)
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Improvement! Best val_loss: {val_loss:.4f}")

        else:
            # No improvement: increment counter
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  [EarlyStopping] Stopping training. Best val_loss: {self.best_loss:.4f}")

        return self.early_stop

    def _save_checkpoint(self, model: nn.Module, val_loss: float, val_acc: float) -> None:
        """Save model checkpoint when validation improves."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"  [Checkpoint] Best model saved → {self.save_path} (loss={val_loss:.4f}, acc={val_acc:.4f})")


# =============================================================================
# METRIC UTILITIES
# =============================================================================

def compute_metrics(
    all_labels: List[int],
    all_preds: List[int],
) -> Dict[str, float]:
    """
    Compute classification metrics from prediction lists.

    Args:
        all_labels: Ground truth integer labels
        all_preds:  Predicted integer labels

    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds,  average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds,       average="weighted", zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# =============================================================================
# TRAINING & VALIDATION LOOPS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    epoch: int,
    accumulation_steps: int = 1,
) -> Tuple[float, float]:
    """
    Train model for one complete epoch.

    Args:
        model:              The neural network
        loader:             Training DataLoader
        criterion:          Loss function (CrossEntropyLoss)
        optimizer:          Optimizer (Adam/AdamW)
        device:             Compute device (cuda/cpu)
        logger:             Logger instance
        epoch:              Current epoch number (for display)
        accumulation_steps: Gradient accumulation steps (for large batches)

    Returns:
        avg_loss: Mean training loss for this epoch
        avg_acc:  Mean training accuracy for this epoch
    """
    model.train()             # set model to training mode (enables dropout, BN)

    total_loss = 0.0          # accumulate batch losses
    all_labels = []           # collect ground truth labels
    all_preds = []            # collect predicted labels
    num_batches = len(loader)

    optimizer.zero_grad()     # clear gradients at epoch start

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to GPU/CPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass ─────────────────────────────────────────────────
        logits = model(images)               # (B, 4) raw class scores

        # ── Compute loss ─────────────────────────────────────────────────
        loss = criterion(logits, labels)

        # ── Gradient accumulation (helps simulate larger batch sizes) ──
        loss = loss / accumulation_steps     # scale loss

        # ── Backward pass ────────────────────────────────────────────────
        loss.backward()

        # ── Optimizer step (every `accumulation_steps` batches) ──────────
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # ── Collect metrics ───────────────────────────────────────────────
        total_loss += loss.item() * accumulation_steps    # undo scaling for logging
        preds = logits.argmax(dim=1).cpu().numpy()        # predicted class indices
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Print batch progress every 20 batches
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            batch_acc = accuracy_score(all_labels, all_preds)
            logger.info(
                f"  Epoch {epoch} | Batch [{batch_idx+1}/{num_batches}] "
                f"| Loss: {total_loss/(batch_idx+1):.4f} "
                f"| Acc: {batch_acc:.4f}"
            )

    # Compute epoch averages
    avg_loss = total_loss / num_batches
    avg_acc  = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate model on validation or test set.

    @torch.no_grad() disables gradient computation for efficiency.

    Args:
        model:     The neural network
        loader:    Validation/Test DataLoader
        criterion: Loss function
        device:    Compute device

    Returns:
        avg_loss:   Mean loss
        avg_acc:    Mean accuracy
        all_labels: Ground truth labels (for detailed metrics)
        all_preds:  Predicted labels (for detailed metrics)
    """
    model.eval()              # set to evaluation mode (disable dropout/BN)

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass (no gradient computation)
        logits = model(images)

        # Loss
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Predictions
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_acc  = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_acc, all_labels, all_preds


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_curves(history: Dict, save_dir: str = "outputs/") -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        history:  Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curve ──────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=3)
    axes[0].set_title("Training & Validation Loss", fontsize=13, pad=10)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Mark best epoch (lowest val loss)
    best_epoch = np.argmin(history["val_loss"]) + 1
    axes[0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
    axes[0].legend()

    # ── Accuracy curve ───────────────────────────────────────────────────
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=3)
    axes[1].set_title("Training & Validation Accuracy", fontsize=13, pad=10)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7)

    plt.suptitle("Brain Tumor ViT-GRU Training Curves", fontsize=15, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Training curves saved → {save_path}")
    plt.close()


def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    save_dir: str = "outputs/",
    split_name: str = "test",
) -> None:
    """
    Plot and save a normalized confusion matrix.

    Args:
        labels:     Ground truth labels
        preds:      Predicted labels
        save_dir:   Directory to save plot
        split_name: 'val' or 'test' (for filename)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalize

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0], linewidths=0.5, linecolor="gray"
    )
    axes[0].set_title(f"Confusion Matrix (Counts) — {split_name.capitalize()}", fontsize=12)
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("True Label", fontsize=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")

    # Normalized percentages
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1], vmin=0, vmax=1, linewidths=0.5, linecolor="gray"
    )
    axes[1].set_title(f"Confusion Matrix (Normalized) — {split_name.capitalize()}", fontsize=12)
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("True Label", fontsize=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Confusion matrix saved → {save_path}")
    plt.close()


def save_classification_report(
    labels: List[int],
    preds: List[int],
    save_dir: str = "outputs/",
    split_name: str = "test",
) -> None:
    """
    Save sklearn classification report as text and CSV.

    Args:
        labels:     Ground truth labels
        preds:      Predicted labels
        save_dir:   Directory to save reports
        split_name: 'val' or 'test'
    """
    os.makedirs(save_dir, exist_ok=True)

    # Generate text report
    report_str = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(f"\n[Evaluation] Classification Report ({split_name}):\n{report_str}")

    # Save text report
    txt_path = os.path.join(save_dir, f"classification_report_{split_name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Classification Report — {split_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report_str)
    print(f"[Report] Saved → {txt_path}")

    # Generate and save CSV
    report_dict = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    df = pd.DataFrame(report_dict).transpose().round(4)
    csv_path = os.path.join(save_dir, f"classification_report_{split_name}.csv")
    df.to_csv(csv_path)
    print(f"[Report] CSV saved → {csv_path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    data_dir: str,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    save_dir: str = "saved_models/",
    output_dir: str = "outputs/",
    pretrained: bool = True,
    num_workers: int = 2,
    accumulation_steps: int = 1,
) -> None:
    """
    Full training pipeline for the Hybrid ViT-GRU model.

    Args:
        data_dir:           Path to dataset directory
        num_epochs:         Maximum training epochs
        batch_size:         Training batch size
        learning_rate:      Initial learning rate
        weight_decay:       L2 regularization coefficient
        patience:           Early stopping patience (epochs)
        save_dir:           Directory to save model checkpoints
        output_dir:         Directory to save plots and logs
        pretrained:         Use pretrained ViT backbone
        num_workers:        DataLoader worker processes
        accumulation_steps: Gradient accumulation steps
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("  Brain Tumor Detection — Training Started")
    logger.info("=" * 60)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info(f"\nLoading dataset from: {data_dir}")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info(f"Dataset loaded: {dataset_info['total_size']} total images")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("\nCreating Hybrid ViT-GRU model...")
    model = create_model(num_classes=4, pretrained=pretrained, device=str(device))
    get_model_summary(model, device=str(device))

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing helps generalization

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine annealing: smoothly reduces LR from lr to 0 over training
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    # Early stopping
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    early_stopper = EarlyStopping(
        patience=patience,
        verbose=True,
        save_path=checkpoint_path,
    )

    # ── Training History ──────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"  Optimizer   : AdamW (lr={learning_rate}, wd={weight_decay})")
    logger.info(f"  Scheduler   : CosineAnnealingLR")
    logger.info(f"  Early stop  : patience={patience}")
    logger.info("-" * 60)

    start_time = time.time()

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch [{epoch}/{num_epochs}] | LR: {current_lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch, accumulation_steps
        )

        # ── Validate ───────────────────────────────────────────────────
        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )

        # ── Update scheduler ───────────────────────────────────────────
        scheduler.step()

        # ── Record history ─────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # ── Log epoch summary ──────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        logger.info(
            f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}\n"
            f"  Val   → Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}\n"
            f"  Time: {epoch_time:.1f}s"
        )

        # ── Early stopping check ───────────────────────────────────────
        should_stop = early_stopper(val_loss, val_acc, model)
        if should_stop:
            logger.info(f"\n[Training] Early stopping triggered at epoch {epoch}")
            break

    # ── Training Complete ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best val accuracy: {early_stopper.best_acc:.4f}")
    logger.info(f"Best val loss:     {early_stopper.best_loss:.4f}")
    logger.info(f"Best model saved → {checkpoint_path}")

    # ── Save training curves ──────────────────────────────────────────────
    plot_training_curves(history, save_dir=output_dir)

    # ── Save training history CSV ─────────────────────────────────────────
    history_df = pd.DataFrame({
        "epoch":      list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "val_loss":   history["val_loss"],
        "train_acc":  history["train_acc"],
        "val_acc":    history["val_acc"],
        "lr":         history["lr"],
    })
    history_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved → {history_path}")

    # ── Final evaluation on test set ──────────────────────────────────────
    logger.info(f"\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc, test_labels, test_preds = validate(
        model, test_loader, criterion, device
    )
    test_metrics = compute_metrics(test_labels, test_preds)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SET EVALUATION")
    logger.info(f"{'='*60}")
    logger.info(f"  Loss      : {test_loss:.4f}")
    logger.info(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {test_metrics['precision']:.4f}")
    logger.info(f"  Recall    : {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score  : {test_metrics['f1']:.4f}")

    # ── Save confusion matrix and classification report ───────────────────
    plot_confusion_matrix(test_labels, test_preds, save_dir=output_dir, split_name="test")
    save_classification_report(test_labels, test_preds, save_dir=output_dir, split_name="test")

    # ── Save final model with metadata ────────────────────────────────────
    final_save_path = os.path.join(save_dir, "final_model.pth")
    torch.save({
        "model_state_dict":  model.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "epoch":             epoch,
        "best_val_acc":      early_stopper.best_acc,
        "best_val_loss":     early_stopper.best_loss,
        "test_acc":          test_metrics["accuracy"],
        "history":           history,
        "class_names":       CLASS_NAMES,
        "dataset_info":      dataset_info,
        "hyperparameters": {
            "lr":          learning_rate,
            "batch_size":  batch_size,
            "weight_decay": weight_decay,
        },
    }, final_save_path)
    logger.info(f"Final model saved → {final_save_path}")
    logger.info("Training pipeline complete ✓")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line training arguments."""
    parser = argparse.ArgumentParser(
        description="Train Brain Tumor ViT-GRU Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",    type=str, default="dataset/Training",  help="Dataset directory path")
    parser.add_argument("--epochs",      type=int, default=30,                  help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int, default=32,                  help="Training batch size")
    parser.add_argument("--lr",          type=float, default=1e-4,              help="Learning rate")
    parser.add_argument("--weight_decay",type=float, default=1e-4,              help="Weight decay (L2)")
    parser.add_argument("--patience",    type=int, default=10,                  help="Early stopping patience")
    parser.add_argument("--save_dir",    type=str, default="saved_models/",     help="Checkpoint save directory")
    parser.add_argument("--output_dir",  type=str, default="outputs/",          help="Outputs/plots directory")
    parser.add_argument("--workers",     type=int, default=2,                   help="DataLoader workers")
    parser.add_argument("--no_pretrain", action="store_true",                   help="Disable pretrained ViT")
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrain,
        num_workers=args.workers,
    )
