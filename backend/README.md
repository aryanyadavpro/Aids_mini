# ⚡ ResoGenre Backend API

> **Self-Contained FastAPI + PyTorch Music Genre Classification Service**  
> Ready for direct deployment on Render, Railway, Hugging Face Spaces, or Docker.

---

## 📁 Directory Contents

- `main.py`: FastAPI server with CORS, model lifespan management, and routing.
- `routes.py`: Endpoints (`/predict`, `/waveform`, `/spectrogram`, `/health`, `/sample/{genre}`, `/samples`).
- `model.py`: 4-Stage 2D CNN architecture (`GenreCNN`).
- `features.py`: Audio loading (22,050 Hz Mono), peak normalization, 19-slice segmentation, Log-Mel Spectrogram extraction.
- `schemas.py`: Pydantic request and response schemas.
- `utils.py`: CUDA/CPU device selector and checkpoint loader.
- `config.py`: Centralized audio and server parameters.
- `models/resogenre_cnn_best.pth`: Pinned best model checkpoint weights (5.1 MB).
- `samples/`: Lightweight audio samples for instant 1-click UI testing.
- `Dockerfile`: Production Docker container definition.
- `requirements.txt`: Python package dependencies.

---

## 🚀 Running Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation will be available at:
👉 `http://localhost:8000/docs`

---

## 🌐 Deploying to Cloud (Render / Railway / Hugging Face)

### Option 1: Render.com (Free)
1. Go to [Render.com](https://render.com) -> **New Web Service**.
2. Connect your GitHub repository.
3. In settings:
   - **Root Directory:** `backend`
   - **Runtime:** **Docker** (or Python 3: Build Command: `pip install -r requirements.txt`, Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`)
4. Click **Deploy**!

### Option 2: Railway.app
1. Go to [Railway.app](https://railway.app) -> **New Project** -> **Deploy from GitHub**.
2. In service settings, set **Root Directory** to `/backend`.
3. Generate domain under **Networking**.

### Option 3: Docker Run
```bash
cd backend
docker build -t resogenre-backend .
docker run -p 8000:8000 resogenre-backend
```
