"""
config.py — Global Hyperparameters & Constants for ResoGenre.

Centralizes all configuration values used across the ML pipeline,
FastAPI backend, and data processing modules.
"""

import os

# Limit BLAS threads to prevent Windows OpenBLAS memory allocation failures
for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[var] = "4"

from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "Music_data" / "Data" / "genres_original"
if not RAW_DATA_DIR.exists():
    RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# Audio Processing
# ============================================================
SAMPLE_RATE: int = 22_050
SEGMENT_DURATION: float = 3.0  # seconds
SEGMENT_SAMPLES: int = int(SAMPLE_RATE * SEGMENT_DURATION)  # 66,150
NUM_SEGMENTS_PER_TRACK: int = 10  # 30s / 3s

# Mel-Spectrogram Parameters
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 128

# ============================================================
# Dataset
# ============================================================
GENRES: list[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
NUM_CLASSES: int = len(GENRES)
GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE: dict[int, str] = {i: g for i, g in enumerate(GENRES)}

# ============================================================
# Data Splitting (at song level — leakage prevention)
# ============================================================
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1

# ============================================================
# Training Hyperparameters
# ============================================================
RANDOM_SEED: int = 42
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 20
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EARLY_STOPPING_PATIENCE: int = 7

# Dropout rates
DROPOUT_CONV: float = 0.25
DROPOUT_CONV_DEEP: float = 0.3
DROPOUT_FC: float = 0.5

# ============================================================
# Baseline Model
# ============================================================
RF_N_ESTIMATORS: int = 200
NUM_MFCC: int = 20  # 20 MFCCs → 40 features (mean + var)
NUM_CHROMA: int = 12  # 12 chroma pitch classes

# ============================================================
# Performance Targets
# ============================================================
TARGET_SEGMENT_ACCURACY: float = 0.82
TARGET_TRACK_ACCURACY: float = 0.87
MAX_INFERENCE_LATENCY_MS: int = 500

# ============================================================
# Model Checkpoint Naming
# ============================================================
BEST_MODEL_FILENAME: str = "resogenre_cnn_best.pth"
BASELINE_MODEL_FILENAME: str = "resogenre_rf_baseline.pkl"

# ============================================================
# FastAPI / Server
# ============================================================
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
FRONTEND_URL: str = "http://localhost:3000"
ALLOWED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".ogg"}
