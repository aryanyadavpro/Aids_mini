"""
backend/config.py — Self-Contained Backend Configuration for ResoGenre.

Provides centralized configuration for the FastAPI service, model loading,
audio preprocessing, and inference without depending on files outside backend/.
"""

import os
from pathlib import Path

# Limit CPU BLAS threads to prevent memory allocation spikes
for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[var] = "4"

# ============================================================
# Paths
# ============================================================
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
BEST_MODEL_FILENAME = "resogenre_cnn_best.pth"
SAMPLES_DIR = BACKEND_DIR / "samples"

# ============================================================
# Audio Preprocessing Parameters
# ============================================================
SAMPLE_RATE: int = 22_050
SEGMENT_DURATION: float = 3.0  # seconds
SEGMENT_SAMPLES: int = int(SAMPLE_RATE * SEGMENT_DURATION)  # 66,150
HOP_DURATION: float = 1.5      # 50% overlap for dense sliding window

# Mel-Spectrogram Parameters
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 128
TARGET_SPECTROGRAM_LENGTH: int = 130

# ============================================================
# Dataset & Classes
# ============================================================
GENRES: list[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
NUM_CLASSES: int = len(GENRES)
GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE: dict[int, str] = {i: g for i, g in enumerate(GENRES)}

# ============================================================
# CNN Model Architecture Parameters
# ============================================================
DROPOUT_CONV: float = 0.25
DROPOUT_CONV_DEEP: float = 0.30
DROPOUT_FC: float = 0.50

# ============================================================
# Server Configuration
# ============================================================
API_HOST: str = os.environ.get("HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("PORT", "8000"))
ALLOWED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".ogg", ".flac"}
