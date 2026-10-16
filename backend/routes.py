"""
backend/routes.py — REST API Routes for ResoGenre Backend.
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

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from config import (
    GENRES,
    IDX_TO_GENRE,
    SAMPLES_DIR,
    ALLOWED_AUDIO_EXTENSIONS,
)
from features import (
    load_audio,
    segment_audio,
    compute_mel_spectrogram,
    mel_spectrogram_to_tensor,
)
from schemas import (
    PredictionResponse,
    WaveformResponse,
    SpectrogramResponse,
    HealthResponse,
)

logger = logging.getLogger("resogenre")

router = APIRouter()

# Global reference to model state set during lifespan startup
model_holder = {
    "model": None,
    "device": None,
    "loaded": False,
}


def set_model_state(model, device, loaded: bool):
    """Set global model state reference."""
    model_holder["model"] = model
    model_holder["device"] = device
    model_holder["loaded"] = loaded


def _save_upload_to_temp(upload: UploadFile) -> Path:
    """Save uploaded audio file to temporary location."""
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

    Extracts 19 overlapping 3s slices, runs batched GPU forward pass,
    and returns soft-probability average and majority-vote consensus.
    """
    temp_path = _save_upload_to_temp(file)

    try:
        start_time = time.time()
        model = model_holder["model"]
        device = model_holder["device"]

        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

        # Load and segment audio with 50% overlap (1.5s hop)
        y, sr = load_audio(str(temp_path))
        segments = segment_audio(y, hop_duration=1.5)

        # Batch inference on GPU/device
        tensors = []
        for seg in segments:
            S = compute_mel_spectrogram(seg)
            tensors.append(mel_spectrogram_to_tensor(S, target_length=130))
        batch_tensor = torch.stack(tensors).to(device)  # [N, 1, 128, 130]

        with torch.no_grad():
            outputs = model(batch_tensor)  # log_softmax [N, 10]
            probs = torch.exp(outputs)      # [N, 10]
            avg_probs = probs.mean(dim=0).cpu().numpy()  # [10]

            _, segment_preds_idx = outputs.max(1)
            segment_preds = [IDX_TO_GENRE[idx.item()] for idx in segment_preds_idx]

        # Soft voting genre (argmax of averaged probabilities)
        predicted_idx = int(np.argmax(avg_probs))
        predicted_genre = IDX_TO_GENRE[predicted_idx]
        confidence = float(avg_probs[predicted_idx])

        # Majority vote genre
        vote_counts = Counter(segment_preds)
        majority_genre = vote_counts.most_common(1)[0][0]

        probabilities = {
            genre: float(avg_probs[idx])
            for idx, genre in IDX_TO_GENRE.items()
        }

        inference_time_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            predicted_genre=predicted_genre,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            segment_predictions=segment_preds,
            majority_vote_genre=majority_genre,
            inference_time_ms=round(inference_time_ms, 2),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/waveform", response_model=WaveformResponse)
async def get_waveform(
    file: UploadFile = File(...),
    num_points: int = 200,
):
    """Downsample audio waveform to 200 points for canvas rendering."""
    temp_path = _save_upload_to_temp(file)

    try:
        y, sr = load_audio(str(temp_path))
        duration = float(len(y) / sr)

        chunk_size = max(1, len(y) // num_points)
        downsampled = []
        for i in range(num_points):
            start = i * chunk_size
            end = min(start + chunk_size, len(y))
            if start < len(y):
                chunk = y[start:end]
                peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
                downsampled.append(round(peak, 4))
            else:
                downsampled.append(0.0)

        return WaveformResponse(
            samples=downsampled,
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
    """Return Mel-spectrogram as a base64-encoded PNG image."""
    temp_path = _save_upload_to_temp(file)

    try:
        y, sr = load_audio(str(temp_path))

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)

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


@router.get("/samples")
async def get_sample_tracks():
    """List sample audio tracks available for 1-click testing."""
    samples = []
    sample_genres = ["classical", "metal", "blues", "jazz", "disco", "pop", "reggae", "rock", "country", "hiphop"]
    for genre in sample_genres:
        sample_file = SAMPLES_DIR / f"{genre}.wav"
        if sample_file.exists():
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

    sample_path = SAMPLES_DIR / f"{genre}.wav"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file for genre {genre} not found")

    return FileResponse(
        path=str(sample_path),
        media_type="audio/wav",
        filename=sample_path.name,
    )
