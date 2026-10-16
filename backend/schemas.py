"""
backend/schemas.py — Pydantic Request & Response Models for ResoGenre.
"""

from pydantic import BaseModel


class GenrePrediction(BaseModel):
    """Single genre prediction item."""
    genre: str
    confidence: float


class PredictionResponse(BaseModel):
    """Prediction response returned by POST /predict."""
    predicted_genre: str
    confidence: float
    probabilities: dict[str, float]
    segment_predictions: list[str]
    majority_vote_genre: str
    inference_time_ms: float


class WaveformResponse(BaseModel):
    """Waveform amplitudes returned by POST /waveform."""
    samples: list[float]
    sample_rate: int
    duration: float


class SpectrogramResponse(BaseModel):
    """Base64-encoded Mel-spectrogram image returned by POST /spectrogram."""
    image_base64: str
    width: int
    height: int


class HealthResponse(BaseModel):
    """Status returned by GET /health."""
    status: str
    model_loaded: bool
    device: str
