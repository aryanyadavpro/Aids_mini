"""
api/routes.py — REST API Routes for ResoGenre.

Provides endpoints for genre classification, waveform/spectrogram
visualization, dataset samples, and system health checks.
"""

import io
import time
import base64
import logging
import tempfile
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

from config import (
    GENRES,
    IDX_TO_GENRE,
    RAW_DATA_DIR,
    ALLOWED_AUDIO_EXTENSIONS,
)
from src.features import (
    load_audio,
    segment_audio,
    compute_mel_spectrogram,
    mel_spectrogram_to_tensor,
)
from api.schemas import (
    PredictionResponse,
    WaveformResponse,
    SpectrogramResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global reference to model state passed from main.py lifespan
model_holder = {
    "model": None,
    "device": None,
    "loaded": False,
}


def set_model_state(model, device, loaded: bool):
    """Set model state references for routing."""
    model_holder["model"] = model
    model_holder["device"] = device
    model_holder["loaded"] = loaded


def _save_upload_to_temp(upload: UploadFile) -> Path:
    """Save uploaded file to a temporary location."""
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Supported: {ALLOWED_AUDIO_EXTENSIONS}",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload.file.read()
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_holder["loaded"],
        device=str(model_holder["device"]) if model_holder["device"] else "none",
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_genre(file: UploadFile = File(...)):
    """Predict the genre of an uploaded audio file.

    Segments the audio into 3s slices, predicts each segment using the 2D CNN,
    and returns majority-vote genre with averaged confidence distribution.
    """
    temp_path = _save_upload_to_temp(file)

    try:
        start_time = time.time()
        model = model_holder["model"]
        device = model_holder["device"]

        if model is None:
            raise HTTPException(status_code=503, detail="Model is not currently loaded.")

        # Load and segment audio with 50% overlap for dense, robust voting
        y, sr = load_audio(str(temp_path))
        segments = segment_audio(y, hop_duration=1.5)

        # Batch inference on all segments at once on GPU/device
        tensors = []
        for seg in segments:
            S = compute_mel_spectrogram(seg)
            tensors.append(mel_spectrogram_to_tensor(S, target_length=130))
        batch_tensor = torch.stack(tensors).to(device)  # [N, 1, 128, 130]

        with torch.no_grad():
            outputs = model(batch_tensor)
            all_probs = torch.exp(outputs).cpu().numpy()  # [N, 10]

        segment_predictions = [IDX_TO_GENRE[int(np.argmax(p))] for p in all_probs]

        # Majority vote across dense segments
        vote_counts = Counter(segment_predictions)
        majority_genre = vote_counts.most_common(1)[0][0]

        # Average probabilities across all dense segments (soft voting)
        avg_probs = np.mean(all_probs, axis=0)
        probabilities = {genre: float(avg_probs[i]) for i, genre in enumerate(GENRES)}

        # Top prediction
        top_idx = int(np.argmax(avg_probs))
        predicted_genre = IDX_TO_GENRE[top_idx]
        confidence = float(avg_probs[top_idx])

        inference_time = (time.time() - start_time) * 1000  # ms

        return PredictionResponse(
            predicted_genre=predicted_genre,
            confidence=confidence,
            probabilities=probabilities,
            segment_predictions=segment_predictions,
            majority_vote_genre=majority_genre,
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/waveform", response_model=WaveformResponse)
async def get_waveform(file: UploadFile = File(...)):
    """Return downsampled waveform data for smooth interactive visualization."""
    temp_path = _save_upload_to_temp(file)

    try:
        y, sr = load_audio(str(temp_path))
        duration = len(y) / sr

        # Downsample for visualization (max ~2000 points)
        target_points = 2000
        if len(y) > target_points:
            step = len(y) // target_points
            y_down = y[::step].tolist()
        else:
            y_down = y.tolist()

        return WaveformResponse(
            samples=y_down,
            sample_rate=sr,
            duration=round(duration, 2),
        )

    except Exception as e:
        logger.error(f"Waveform extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/spectrogram", response_model=SpectrogramResponse)
async def get_spectrogram(file: UploadFile = File(...)):
    """Return mel-spectrogram as a base64-encoded PNG image."""
    temp_path = _save_upload_to_temp(file)

    try:
        y, sr = load_audio(str(temp_path))

        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Generate spectrogram plot
        fig, ax = plt.subplots(figsize=(12, 4))
        librosa.display.specshow(
            S_db, sr=sr, hop_length=512,
            x_axis="time", y_axis="mel",
            ax=ax, cmap="magma",
        )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(pad=0)

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return SpectrogramResponse(
            image_base64=image_base64,
            width=1200,
            height=400,
        )

    except Exception as e:
        logger.error(f"Spectrogram generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)


REPRESENTATIVE_SAMPLES: dict[str, str] = {
    "jazz": "jazz.00005.wav",
    "classical": "classical.00000.wav",
    "metal": "metal.00000.wav",
    "blues": "blues.00000.wav",
    "disco": "disco.00000.wav",
    "pop": "pop.00000.wav",
    "reggae": "reggae.00000.wav",
    "rock": "rock.00000.wav",
}


@router.get("/samples")
async def get_sample_tracks():
    """List available sample audio tracks for 1-click testing."""
    samples = []
    sample_genres = ["classical", "metal", "blues", "jazz", "disco", "pop", "reggae", "rock"]
    for genre in sample_genres:
        genre_dir = RAW_DATA_DIR / genre
        if genre_dir.exists():
            target_name = REPRESENTATIVE_SAMPLES.get(genre)
            sample_file = None
            if target_name and (genre_dir / target_name).exists():
                sample_file = genre_dir / target_name
            else:
                wav_files = sorted(genre_dir.glob("*.wav"))
                if wav_files:
                    sample_file = wav_files[0]
            if sample_file:
                samples.append({
                    "genre": genre,
                    "filename": sample_file.name,
                    "url": f"/sample/{genre}",
                })
    return {"samples": samples}


@router.get("/sample/{genre}")
async def get_sample_audio(genre: str):
    """Stream a sample audio file for the specified genre."""
    genre = genre.lower().strip()
    if genre not in GENRES:
        raise HTTPException(status_code=404, detail=f"Genre {genre} not found")

    genre_dir = RAW_DATA_DIR / genre
    if not genre_dir.exists():
        raise HTTPException(status_code=404, detail="Genre folder not found")

    target_name = REPRESENTATIVE_SAMPLES.get(genre)
    if target_name and (genre_dir / target_name).exists():
        sample_path = genre_dir / target_name
    else:
        wav_files = sorted(genre_dir.glob("*.wav"))
        if not wav_files:
            raise HTTPException(status_code=404, detail="No audio file found for genre")
        sample_path = wav_files[0]

    return FileResponse(
        path=str(sample_path),
        media_type="audio/wav",
        filename=sample_path.name,
    )
