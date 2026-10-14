"""
api/main.py — FastAPI Application for ResoGenre.

Configures application lifespan, model loading, CORS middleware,
and mounts REST API routes.
"""

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    MODELS_DIR,
    BEST_MODEL_FILENAME,
    FRONTEND_URL,
    API_HOST,
    API_PORT,
)
from src.model import GenreCNN
from src.utils import get_device, setup_logging
from api.routes import router, set_model_state

logger = logging.getLogger(__name__)

# Global state for loaded model
model_state = {
    "model": None,
    "device": None,
    "loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    setup_logging()
    device = get_device()
    model = GenreCNN()

    model_path = MODELS_DIR / BEST_MODEL_FILENAME
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded successfully from {model_path}")
        model_state["loaded"] = True
    else:
        logger.warning(f"No model checkpoint found at {model_path}. Using untrained model fallback.")
        model_state["loaded"] = False

    model = model.to(device)
    model.eval()
    model_state["model"] = model
    model_state["device"] = device

    set_model_state(model, device, model_state["loaded"])

    yield

    # Cleanup
    model_state["model"] = None
    model_state["device"] = None
    set_model_state(None, None, False)


app = FastAPI(
    title="ResoGenre API",
    description="Music Genre Classification API powered by a 4-stage 2D CNN",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
