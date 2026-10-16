"""
backend/features.py — Audio Preprocessing & Feature Extraction for ResoGenre.

Converts raw audio (.wav, .mp3) into Log-Mel Spectrogram tensors matching
the 4-Stage 2D CNN input dimension [1, 128, 130].
"""

import logging
from typing import Optional

import librosa
import numpy as np
import torch

from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    SEGMENT_DURATION,
    SEGMENT_SAMPLES,
    HOP_DURATION,
    TARGET_SPECTROGRAM_LENGTH,
)

logger = logging.getLogger(__name__)


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file, resample to mono, and dynamic-peak normalize.

    Args:
        filepath: Path to audio file.
        sr: Target sampling rate (defaults to 22,050 Hz).

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    try:
        y, sr_out = librosa.load(filepath, sr=sr, mono=True)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = (y / peak) * 0.95
        return y.astype(np.float32), sr_out
    except Exception as e:
        logger.error(f"Failed to load audio: {filepath} — {e}")
        raise


def segment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION,
    hop_duration: Optional[float] = HOP_DURATION,
) -> list[np.ndarray]:
    """Segment an audio clip into 3s slices with 50% sliding-window hop.

    Args:
        y: 1D audio time-series array.
        sr: Sample rate.
        segment_duration: Slice duration in seconds (3.0s).
        hop_duration: Step between slices in seconds (1.5s).

    Returns:
        List of audio segments as 1D arrays of length `segment_samples`.
    """
    segment_samples = int(sr * segment_duration)
    hop_samples = int(sr * hop_duration) if hop_duration is not None else segment_samples

    total_samples = len(y)

    # If audio is shorter than 3s, zero-pad it
    if total_samples < segment_samples:
        padded = np.zeros(segment_samples, dtype=y.dtype)
        padded[:total_samples] = y
        return [padded]

    segments = []
    start = 0
    while start + segment_samples <= total_samples:
        seg = y[start : start + segment_samples]
        segments.append(seg)
        start += hop_samples

    # If remainder left at end, take the final segment
    if len(segments) == 0:
        padded = np.zeros(segment_samples, dtype=y.dtype)
        padded[:total_samples] = y
        segments.append(padded)

    return segments


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Compute Log-Mel Spectrogram in decibels.

    Args:
        y: 1D audio segment array.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_mels: Number of Mel frequency bands.

    Returns:
        2D array of shape [n_mels, time_steps] in dB scale.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def mel_spectrogram_to_tensor(
    S: np.ndarray,
    target_length: int = TARGET_SPECTROGRAM_LENGTH,
) -> torch.Tensor:
    """Normalize dB values to [0, 1] and pad/truncate to target length.

    Args:
        S: Mel-spectrogram array [n_mels, time_steps].
        target_length: Fixed number of time frames (130).

    Returns:
        PyTorch FloatTensor of shape [1, n_mels, target_length].
    """
    s_min = S.min()
    s_max = S.max()
    denom = s_max - s_min
    if denom > 1e-8:
        S_norm = (S - s_min) / denom
    else:
        S_norm = np.zeros_like(S)

    n_mels, time_steps = S_norm.shape
    if time_steps < target_length:
        pad_width = target_length - time_steps
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode="constant")
    elif time_steps > target_length:
        S_norm = S_norm[:, :target_length]

    tensor = torch.from_numpy(S_norm).float().unsqueeze(0)  # [1, 128, 130]
    return tensor


def audio_to_mel_tensor(
    filepath: str,
    sr: int = SAMPLE_RATE,
    hop_duration: Optional[float] = HOP_DURATION,
) -> list[torch.Tensor]:
    """End-to-end pipeline: file path -> list of [1, 128, 130] tensors."""
    y, sr = load_audio(filepath, sr=sr)
    segments = segment_audio(y, sr=sr, hop_duration=hop_duration)
    tensors = []
    for seg in segments:
        S = compute_mel_spectrogram(seg, sr=sr)
        tensor = mel_spectrogram_to_tensor(S)
        tensors.append(tensor)
    return tensors
