"""
backend/main.py — FastAPI Application for ResoGenre.

Production entrypoint for the ResoGenre music genre classification backend.
Can be deployed directly as the root of a backend service (e.g. on Render,
Railway, Hugging Face Spaces, or Docker).
"""

import os
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    MODELS_DIR,
    BEST_MODEL_FILENAME,
    API_HOST,
    API_PORT,
)
from model import GenreCNN
from utils import get_device, setup_logging
from routes import router, set_model_state

logger = logging.getLogger("resogenre")

model_state = {
    "model": None,
    "device": None,
    "loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load PyTorch CNN model on startup into VRAM/RAM."""
    setup_logging()
    device = get_device()
    model = GenreCNN()

    model_path = MODELS_DIR / BEST_MODEL_FILENAME
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Model successfully loaded from {model_path} onto {device}")
        model_state["loaded"] = True
    else:
        logger.warning(f"No checkpoint found at {model_path}! Inference will use fallback initialization.")
        model_state["loaded"] = False

    model = model.to(device)
    model.eval()
    model_state["model"] = model
    model_state["device"] = device

    set_model_state(model, device, model_state["loaded"])

    yield

    # Cleanup on shutdown
    model_state["model"] = None
    model_state["device"] = None
    set_model_state(None, None, False)


app = FastAPI(
    title="ResoGenre API",
    description="Deep-Learning Music Genre Classification API powered by a 4-Stage 2D CNN",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration — Allow all origins so any deployed Vercel domain can communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return JSONResponse({
        "service": "ResoGenre API",
        "status": "online",
        "model_loaded": model_state["loaded"],
        "device": str(model_state["device"]),
        "docs_url": "/docs",
    })

# Mount REST endpoints (/predict, /waveform, /spectrogram, /health, /samples, /sample/{genre})
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
