"""
src/train.py — Training Loop for ResoGenre CNN.

Handles training, validation, checkpoint saving,
learning rate scheduling, and early stopping.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    BEST_MODEL_FILENAME,
    OUTPUTS_DIR,
)
from src.utils import save_checkpoint, get_device

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The CNN model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)        # [B, 1, 128, 130]
        targets = targets.to(device)      # [B]

        optimizer.zero_grad()
        outputs = model(inputs)           # [B, 10] log-softmax
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            logger.info(
                f"  Batch {batch_idx + 1:3d}/{len(loader)} | "
                f"Loss: {running_loss / total:.4f} | "
                f"Acc: {correct / total:.4f}"
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on validation/test set.

    Args:
        model: The CNN model.
        loader: Validation/Test DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: Optional[torch.device] = None,
) -> dict:
    """Full training pipeline with validation, scheduling, and early stopping.

    Args:
        model: The CNN model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Maximum number of epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        patience: Early stopping patience (epochs without improvement).
        device: Compute device. Auto-detected if None.

    Returns:
        Dictionary with training history (losses, accuracies).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_epoch = 1

    ckpt_path = OUTPUTS_DIR.parent / "models" / BEST_MODEL_FILENAME
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("val_accuracy", 0.0)
            logger.info(f"Resuming training from epoch {start_epoch} (best val acc: {best_val_acc:.4f})")
        except Exception as e:
            logger.warning(f"Could not resume from checkpoint: {e}")

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                filename=BEST_MODEL_FILENAME,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    total_time = time.time() - start_time
    logger.info(
        f"Training complete in {total_time:.1f}s | "
        f"Best val accuracy: {best_val_acc:.4f}"
    )

    history["best_val_acc"] = best_val_acc
    history["total_time_seconds"] = total_time
    return history


if __name__ == "__main__":
    import json
    from src.utils import setup_logging, ensure_dirs, set_seed, load_checkpoint
    from src.dataset import create_dataloaders, get_song_paths, split_songs
    from src.model import GenreCNN
    from src.evaluate import (
        get_predictions,
        compute_metrics,
        plot_confusion_matrix,
        plot_training_history,
        majority_vote_accuracy,
    )

    setup_logging()
    ensure_dirs()
    set_seed()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Initializing datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders()

    # Initialize model
    model = GenreCNN().to(device)
    logger.info("Starting CNN training...")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
    )

    # Plot training curves
    plot_training_history(history)

    # Load best model checkpoint for evaluation
    best_model = GenreCNN().to(device)
    load_checkpoint(best_model, filename=BEST_MODEL_FILENAME, device=device)

    # Evaluate on test set
    logger.info("Evaluating best model on test set...")
    y_true, y_pred, y_probs = get_predictions(best_model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # Majority-vote accuracy on full 30s tracks
    songs = get_song_paths()
    _, _, test_songs = split_songs(songs)
    maj_acc = majority_vote_accuracy(best_model, test_songs, device)
    metrics["majority_vote_track_accuracy"] = maj_acc

    # Save metrics JSON
    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        # Convert non-serializable objects
        json.dump(
            {
                "test_segment_accuracy": float(metrics["accuracy"]),
                "test_macro_f1": float(metrics["macro_f1"]),
                "test_weighted_f1": float(metrics["weighted_f1"]),
                "majority_vote_track_accuracy": float(maj_acc),
                "best_val_accuracy": float(history["best_val_acc"]),
                "total_train_time_seconds": float(history["total_time_seconds"]),
            },
            f,
            indent=2,
        )
    logger.info(f"Evaluation metrics saved to {metrics_path}")

