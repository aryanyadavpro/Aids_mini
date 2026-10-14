"""
src/dataset.py — PyTorch Dataset & DataLoader for ResoGenre.

Implements leakage-safe data splitting at the song level,
audio segmentation, and mel-spectrogram generation as a
PyTorch Dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    GENRES,
    GENRE_TO_IDX,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_SEGMENTS_PER_TRACK,
)
from src.features import (
    load_audio,
    segment_audio,
    compute_mel_spectrogram,
    mel_spectrogram_to_tensor,
)

logger = logging.getLogger(__name__)


def get_song_paths(data_dir: Path = RAW_DATA_DIR) -> list[tuple[Path, str]]:
    """Collect all song file paths with their genre labels.

    Args:
        data_dir: Root directory containing genre subdirectories.

    Returns:
        List of (filepath, genre_label) tuples.
    """
    songs = []
    for genre in GENRES:
        genre_dir = data_dir / genre
        if not genre_dir.exists():
            logger.warning(f"Genre directory not found: {genre_dir}")
            continue
        for audio_file in sorted(genre_dir.glob("*.*")):
            if audio_file.suffix.lower() in {".wav", ".mp3", ".ogg"}:
                if audio_file.name == "jazz.00054.wav":
                    continue
                songs.append((audio_file, genre))
    logger.info(f"Found {len(songs)} songs across {len(GENRES)} genres")
    return songs


def split_songs(
    songs: list[tuple[Path, str]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    """Split songs into train/val/test at the SONG level (leakage prevention).

    This ensures that 3s segments from the same 30s track never
    appear across different partitions.

    Args:
        songs: List of (filepath, genre) tuples.
        train_ratio: Fraction for training. Default 0.8.
        val_ratio: Fraction for validation. Default 0.1.
        test_ratio: Fraction for testing. Default 0.1.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_songs, val_songs, test_songs).
    """
    paths = [s[0] for s in songs]
    labels = [s[1] for s in songs]

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_labels,
    )

    train_songs = list(zip(train_paths, train_labels))
    val_songs = list(zip(val_paths, val_labels))
    test_songs = list(zip(test_paths, test_labels))

    logger.info(
        f"Song-level split — Train: {len(train_songs)}, "
        f"Val: {len(val_songs)}, Test: {len(test_songs)}"
    )
    return train_songs, val_songs, test_songs


class GenreDataset(Dataset):
    """PyTorch Dataset for mel-spectrogram-based genre classification.

    Each item is a 3s segment mel-spectrogram tensor paired with
    its genre label index.

    Attributes:
        segments: List of (mel_tensor, label_index) tuples.
    """

    def __init__(
        self,
        songs: list[tuple[Path, str]],
        target_length: int = 130,
        augment: bool = False,
        cache_name: Optional[str] = None,
    ) -> None:
        """Initialize dataset by processing songs into segments.

        Args:
            songs: List of (filepath, genre) tuples.
            target_length: Target time-frame dimension for spectrograms.
            augment: Whether to apply data augmentation (training only).
            cache_name: Optional filename for saving/loading processed segments.
        """
        self.segments: list[tuple[torch.Tensor, int]] = []
        self.target_length = target_length
        self.augment = augment
        self.cache_name = cache_name

        if cache_name and (PROCESSED_DATA_DIR / f"{cache_name}.pt").exists():
            cache_file = PROCESSED_DATA_DIR / f"{cache_name}.pt"
            logger.info(f"Loading cached segments from {cache_file}...")
            self.segments = torch.load(cache_file, weights_only=False)
            logger.info(f"Loaded {len(self.segments)} cached segments")
        else:
            self._process_songs(songs)
            if cache_name:
                PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
                cache_file = PROCESSED_DATA_DIR / f"{cache_name}.pt"
                torch.save(self.segments, cache_file)
                logger.info(f"Saved {len(self.segments)} segments to {cache_file}")

    def _process_songs(self, songs: list[tuple[Path, str]]) -> None:
        """Process all songs into 3s segment mel-spectrograms."""
        total_songs = len(songs)
        for i, (filepath, genre) in enumerate(songs, 1):
            label_idx = GENRE_TO_IDX[genre]
            try:
                y, sr = load_audio(str(filepath))
                audio_segments = segment_audio(y)
                for seg in audio_segments:
                    S = compute_mel_spectrogram(seg)
                    tensor = mel_spectrogram_to_tensor(
                        S, target_length=self.target_length
                    )
                    self.segments.append((tensor, label_idx))
            except Exception as e:
                logger.warning(f"Skipping {filepath}: {e}")
            if i % 100 == 0 or i == total_songs:
                logger.info(f"Processed {i}/{total_songs} songs ({len(self.segments)} segments)")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor, label = self.segments[idx]
        if self.augment:
            tensor = self._apply_augmentation(tensor)
        return tensor, label

    @staticmethod
    def _apply_augmentation(tensor: torch.Tensor) -> torch.Tensor:
        """Apply simple data augmentation to mel-spectrogram tensor.

        Augmentations:
            - Time masking: zero out a random contiguous block of time frames
            - Frequency masking: zero out a random contiguous block of mel bands

        Args:
            tensor: Input tensor of shape [1, n_mels, time_frames].

        Returns:
            Augmented tensor.
        """
        tensor = tensor.clone()
        _, n_mels, n_frames = tensor.shape

        # Time masking (mask up to 10% of frames)
        if torch.rand(1).item() > 0.5:
            mask_len = int(n_frames * 0.1)
            if mask_len > 0:
                start = torch.randint(0, n_frames - mask_len, (1,)).item()
                tensor[:, :, start : start + mask_len] = 0.0

        # Frequency masking (mask up to 10% of mel bands)
        if torch.rand(1).item() > 0.5:
            mask_len = int(n_mels * 0.1)
            if mask_len > 0:
                start = torch.randint(0, n_mels - mask_len, (1,)).item()
                tensor[:, start : start + mask_len, :] = 0.0

        return tensor


def create_dataloaders(
    data_dir: Path = RAW_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with leakage-safe splitting.

    Args:
        data_dir: Directory containing genre subdirectories.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    songs = get_song_paths(data_dir)
    train_songs, val_songs, test_songs = split_songs(songs)

    train_dataset = GenreDataset(train_songs, augment=True, cache_name="train_segments")
    val_dataset = GenreDataset(val_songs, augment=False, cache_name="val_segments")
    test_dataset = GenreDataset(test_songs, augment=False, cache_name="test_segments")

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    logger.info(
        f"DataLoaders created — Train: {len(train_dataset)} segments, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader
