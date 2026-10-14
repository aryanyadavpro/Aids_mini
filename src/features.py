"""
src/features.py — Audio Feature Extraction for ResoGenre.

Handles mel-spectrogram computation, MFCC extraction,
and handcrafted feature generation for the baseline model.
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
    NUM_SEGMENTS_PER_TRACK,
    NUM_MFCC,
    NUM_CHROMA,
)

logger = logging.getLogger(__name__)


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file, resample, and peak-normalize to standard range.

    Args:
        filepath: Path to the audio file (.wav, .mp3, .ogg).
        sr: Target sample rate. Defaults to 22050 Hz.

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    try:
        y, sr_out = librosa.load(filepath, sr=sr, mono=True)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y.astype(np.float32), sr_out
    except Exception as e:
        logger.error(f"Failed to load audio: {filepath} — {e}")
        raise


def segment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION,
    num_segments: Optional[int] = None,
    hop_duration: Optional[float] = None,
) -> list[np.ndarray]:
    """Segment an audio clip into 3s slices, with optional sliding-window hop.

    Args:
        y: Audio time-series signal.
        sr: Sample rate.
        segment_duration: Duration of each segment in seconds.
        num_segments: Number of segments for disjoint extraction.
        hop_duration: Optional hop duration for dense overlapping windowing (e.g. 1.5s).

    Returns:
        List of audio segments as numpy arrays.
    """
    segment_samples = int(sr * segment_duration)

    if hop_duration is not None and hop_duration > 0:
        hop_samples = int(sr * hop_duration)
        segments = []
        start = 0
        while start + segment_samples <= len(y):
            segments.append(y[start : start + segment_samples])
            start += hop_samples
        if len(segments) == 0:
            segment = np.zeros(segment_samples, dtype=np.float32)
            segment[: len(y)] = y[:segment_samples]
            segments.append(segment)
        return segments

    n_seg = num_segments if num_segments is not None else NUM_SEGMENTS_PER_TRACK
    segments = []
    for i in range(n_seg):
        start = i * segment_samples
        end = start + segment_samples
        if end <= len(y):
            segments.append(y[start:end])
        else:
            segment = np.zeros(segment_samples, dtype=np.float32)
            remaining = y[start:]
            segment[: len(remaining)] = remaining
            segments.append(segment)
    return segments


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Compute a log-scaled mel-spectrogram from an audio signal.

    Args:
        y: Audio time-series signal.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mels: Number of mel frequency bands.

    Returns:
        Normalized mel-spectrogram as 2D numpy array [n_mels, time_frames].
    """
    # Compute power mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # Convert to dB scale
    S_db = librosa.power_to_db(S, ref=np.max)
    # Min-max normalize to [0, 1]
    S_min = S_db.min()
    S_max = S_db.max()
    if S_max - S_min > 0:
        S_norm = (S_db - S_min) / (S_max - S_min)
    else:
        S_norm = np.zeros_like(S_db)
    return S_norm.astype(np.float32)


def mel_spectrogram_to_tensor(
    S: np.ndarray, target_length: Optional[int] = 130
) -> torch.Tensor:
    """Convert mel-spectrogram to a PyTorch tensor with consistent shape.

    Args:
        S: Mel-spectrogram numpy array [n_mels, time_frames].
        target_length: Target number of time frames. Pads or truncates.

    Returns:
        Tensor of shape [1, n_mels, target_length].
    """
    # Pad or truncate time dimension to target_length
    if target_length is not None:
        current_length = S.shape[1]
        if current_length < target_length:
            pad_width = target_length - current_length
            S = np.pad(S, ((0, 0), (0, pad_width)), mode="constant")
        elif current_length > target_length:
            S = S[:, :target_length]

    # Convert to tensor with channel dimension: [1, n_mels, time_frames]
    tensor = torch.FloatTensor(S).unsqueeze(0)
    return tensor


def audio_to_mel_tensor(
    filepath: str,
    sr: int = SAMPLE_RATE,
    target_length: int = 130,
) -> list[torch.Tensor]:
    """Full pipeline: load audio → segment → mel-spectrograms → tensors.

    Args:
        filepath: Path to the audio file.
        sr: Target sample rate.
        target_length: Target time-frame count for each spectrogram.

    Returns:
        List of tensors, one per segment, each of shape [1, 128, 130].
    """
    y, _ = load_audio(filepath, sr=sr)
    segments = segment_audio(y, sr=sr)
    tensors = []
    for seg in segments:
        S = compute_mel_spectrogram(seg, sr=sr)
        tensor = mel_spectrogram_to_tensor(S, target_length=target_length)
        tensors.append(tensor)
    return tensors


def extract_handcrafted_features(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Extract handcrafted audio features for the baseline model.

    Extracts:
        - 20 MFCCs (mean + variance) → 40 features
        - Spectral centroid (mean) → 1 feature
        - Spectral rolloff (mean) → 1 feature
        - Zero-crossing rate (mean) → 1 feature
        - 12 Chroma features (mean) → 12 features
        Total: 55 features (curated subset of 42 can be used)

    Args:
        y: Audio time-series signal.
        sr: Sample rate.

    Returns:
        Feature vector as 1D numpy array.
    """
    features = []

    # MFCCs: mean and variance of 20 coefficients → 40 features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
    features.extend(np.mean(mfccs, axis=1))  # 20 means
    features.extend(np.var(mfccs, axis=1))   # 20 variances

    # Spectral centroid → 1 feature
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    # Spectral rolloff → 1 feature
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))

    # Zero-crossing rate → 1 feature
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    # Chroma features → 12 features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    return np.array(features, dtype=np.float32)
