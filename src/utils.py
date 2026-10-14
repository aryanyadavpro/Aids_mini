"""
src/utils.py — Utility functions for ResoGenre.

Provides seed pinning, device selection, path helpers, and
common utility functions used across the ML pipeline.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import RANDOM_SEED, MODELS_DIR, OUTPUTS_DIR, DATA_DIR

logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Pin all random seeds for reproducibility.

    Args:
        seed: The random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"All random seeds set to {seed}")


def get_device() -> torch.device:
    """Select the best available compute device.

    Returns:
        torch.device: CUDA if available, else CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        device = torch.device("cpu")
        logger.info(f"Using CPU device (threads={num_threads})")
    return device


def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    dirs = [
        MODELS_DIR,
        OUTPUTS_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "processed",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Project directories verified")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the project.

    Args:
        level: Logging level. Defaults to INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model.

    Args:
        model: The PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    filename: Optional[str] = None,
) -> Path:
    """Save a model checkpoint.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        epoch: Current epoch number.
        val_accuracy: Validation accuracy at this checkpoint.
        filename: Optional custom filename. Defaults to epoch-based name.

    Returns:
        Path to the saved checkpoint file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"resogenre_cnn_epoch{epoch}.pth"
    filepath = MODELS_DIR / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
        },
        filepath,
    )
    logger.info(f"Checkpoint saved: {filepath} (val_acc={val_accuracy:.4f})")
    return filepath


def load_checkpoint(
    model: torch.nn.Module,
    filepath: Optional[Path] = None,
    filename: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a model checkpoint.

    Args:
        model: The PyTorch model to load weights into.
        filepath: Optional path to the checkpoint file.
        filename: Optional filename within MODELS_DIR.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Dictionary with checkpoint metadata (epoch, val_accuracy).
    """
    if filepath is None:
        if filename is None:
            filename = "resogenre_cnn_best.pth"
        filepath = MODELS_DIR / filename
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        f"Checkpoint loaded: {filepath} "
        f"(epoch={checkpoint['epoch']}, val_acc={checkpoint['val_accuracy']:.4f})"
    )
    return {
        "epoch": checkpoint["epoch"],
        "val_accuracy": checkpoint["val_accuracy"],
    }
