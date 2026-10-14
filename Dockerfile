# ============================================================
# ResoGenre Backend Dockerfile
# Optimized for Cloud Deployment (Render, Railway, Hugging Face, VPS)
# ============================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install audio processing system dependencies (libsndfile & ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight PyTorch CPU wheel first (saves 2GB image size!)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining python packages
COPY requirements.txt .
# Install without re-installing heavy torch from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, config, and trained model weights
COPY config.py .
COPY src/ ./src/
COPY api/ ./api/
COPY models/resogenre_cnn_best.pth ./models/resogenre_cnn_best.pth
COPY data/.gitkeep ./data/.gitkeep

# Expose port
EXPOSE 8000

# Health check to ensure API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI backend with uvicorn
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
