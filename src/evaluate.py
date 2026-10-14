"""
src/evaluate.py — Evaluation & Visualization for ResoGenre.

Computes classification metrics, generates confusion matrices,
training curves, and per-genre performance breakdowns.
"""

import logging
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from config import GENRES, IDX_TO_GENRE, OUTPUTS_DIR

logger = logging.getLogger(__name__)

# Use a clean style for plots
plt.style.use("seaborn-v0_8-darkgrid")


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get all predictions, true labels, and probabilities from a DataLoader.

    Args:
        model: Trained CNN model.
        loader: DataLoader to evaluate.
        device: Compute device.

    Returns:
        Tuple of (true_labels, predicted_labels, probabilities).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)  # log-softmax
        probs = torch.exp(outputs)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, macro/weighted F1, and per-genre report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(
        y_true, y_pred, target_names=GENRES, output_dict=True
    )
    report_str = classification_report(
        y_true, y_pred, target_names=GENRES
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "classification_report_str": report_str,
    }

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"\n{report_str}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Normalized Confusion Matrix",
) -> None:
    """Generate and save a normalized confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: Path to save the plot. Defaults to outputs dir.
        title: Plot title.
    """
    if save_path is None:
        save_path = OUTPUTS_DIR / "confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=GENRES,
        yticklabels=GENRES,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Genre", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Genre", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Training history dictionary from train_model().
        save_path: Path to save the plot.
    """
    if save_path is None:
        save_path = OUTPUTS_DIR / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=3)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ResoGenre CNN Training Curves", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {save_path}")


def majority_vote_accuracy(
    model: nn.Module,
    test_songs: list,
    device: torch.device,
) -> float:
    """Compute majority-vote ensemble accuracy on full 30s tracks.

    For each song, predict all 10 segments and take the majority genre.

    Args:
        model: Trained CNN model.
        test_songs: List of (filepath, genre) tuples for test songs.
        device: Compute device.

    Returns:
        Majority-vote accuracy on full tracks.
    """
    from src.features import audio_to_mel_tensor
    from config import GENRE_TO_IDX

    model.eval()
    correct = 0
    total = 0

    for filepath, genre in test_songs:
        true_label = GENRE_TO_IDX[genre]
        try:
            tensors = audio_to_mel_tensor(str(filepath))
            segment_preds = []
            for tensor in tensors:
                tensor = tensor.unsqueeze(0).to(device)  # [1, 1, 128, 130]
                output = model(tensor)
                _, predicted = output.max(1)
                segment_preds.append(predicted.item())

            # Majority vote
            vote_counts = Counter(segment_preds)
            majority_genre = vote_counts.most_common(1)[0][0]

            if majority_genre == true_label:
                correct += 1
            total += 1
        except Exception as e:
            logger.warning(f"Skipping {filepath} for majority vote: {e}")

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Majority-vote accuracy on {total} tracks: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    import json
    from src.utils import setup_logging, ensure_dirs, set_seed, get_device, load_checkpoint
    from src.dataset import create_dataloaders, get_song_paths, split_songs
    from src.model import GenreCNN
    from config import BEST_MODEL_FILENAME, MODELS_DIR

    setup_logging()
    ensure_dirs()
    set_seed()

    device = get_device()
    logger.info(f"Evaluating on device: {device}")

    # Load dataloaders (uses cached segments)
    _, _, test_loader = create_dataloaders()

    # Load model
    model = GenreCNN().to(device)
    load_checkpoint(model, filename=BEST_MODEL_FILENAME, device=device)

    # Evaluate segment-level predictions
    logger.info("Computing segment-level test metrics...")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    # Generate Confusion Matrix
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # Compute track-level majority vote accuracy
    logger.info("Computing track-level majority-vote accuracy...")
    songs = get_song_paths()
    _, _, test_songs = split_songs(songs)
    maj_acc = majority_vote_accuracy(model, test_songs, device)

    # Save metrics JSON
    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "test_segment_accuracy": float(metrics["accuracy"]),
                "test_macro_f1": float(metrics["macro_f1"]),
                "test_weighted_f1": float(metrics["weighted_f1"]),
                "majority_vote_track_accuracy": float(maj_acc),
                "baseline_rf_accuracy": 0.6740,
            },
            f,
            indent=2,
        )
    logger.info(f"Evaluation metrics saved to {metrics_path}")

