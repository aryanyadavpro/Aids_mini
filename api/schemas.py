"""
api/schemas.py — Pydantic Models for Request/Response Validation.

Defines the data shapes for all FastAPI endpoints.
"""

from pydantic import BaseModel


class GenrePrediction(BaseModel):
    """Single genre prediction with confidence."""
    genre: str
    confidence: float


class PredictionResponse(BaseModel):
    """Full prediction response from the /predict endpoint."""
    predicted_genre: str
    confidence: float
    probabilities: dict[str, float]
    segment_predictions: list[str]
    majority_vote_genre: str
    inference_time_ms: float


class WaveformResponse(BaseModel):
    """Waveform data for visualization."""
    samples: list[float]
    sample_rate: int
    duration: float


class SpectrogramResponse(BaseModel):
    """Base64-encoded spectrogram image."""
    image_base64: str
    width: int
    height: int


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    device: str
