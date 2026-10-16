"""
backend/utils.py — Utility Functions for ResoGenre Backend.
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn

from config import MODELS_DIR, BEST_MODEL_FILENAME

logger = logging.getLogger("resogenre")


def setup_logging(level: int = logging.INFO) -> None:
    """Set up clean console logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_device() -> torch.device:
    """Select CUDA device if GPU available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        device = torch.device("cpu")
        logger.info(f"Using CPU device (threads={num_threads})")
    return device


def load_checkpoint(
    model: nn.Module,
    filename: str = BEST_MODEL_FILENAME,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load model state dict from checkpoint."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Model checkpoint successfully loaded from {path}")
    return checkpoint
