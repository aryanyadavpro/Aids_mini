# 🎵 ResoGenre — Music Genre Classification System

> Deep-Learning-Based Audio & Music Genre Classification with a Premium Next.js UI

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Next.js](https://img.shields.io/badge/Next.js-15+-black)

## Overview

ResoGenre is an end-to-end ML system that classifies music into 10 genres using:
- **Baseline:** Random Forest on 55 handcrafted audio features (~65-72% accuracy)
- **Primary Model:** 4-stage 2D CNN on mel-spectrograms (≥82% segment accuracy, ≥87% with majority-vote)
- **Frontend:** Premium Next.js UI with animated waveforms, spectrogram heatmaps, and genre prediction cards
- **Backend:** FastAPI serving real-time inference endpoints

## 🎯 Supported Genres

Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock

## 🏗️ Architecture

```
AIDS_mini/
├── src/                    # ML Pipeline (PyTorch)
│   ├── features.py         # Mel-spectrogram & MFCC extraction
│   ├── dataset.py          # PyTorch Dataset with leakage-safe splits
│   ├── model.py            # 4-stage 2D CNN
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Metrics, confusion matrix, training curves
│   └── baseline.py         # Random Forest baseline
├── api/                    # FastAPI Backend
│   ├── main.py             # API server with /predict, /waveform, /spectrogram
│   └── schemas.py          # Pydantic models
├── frontend/               # Next.js Premium UI
├── config.py               # All hyperparameters & constants
└── models/                 # Saved checkpoints
```

## 🚀 Quick Start

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd frontend && npm install
```

### 3. Start the FastAPI Backend
```bash
python -m api.main
```

### 4. Start the Next.js Frontend
```bash
cd frontend && npm run dev
```

### 5. Open the App
Navigate to `http://localhost:3000`

## 📊 Training

```python
from src.utils import set_seed, setup_logging, ensure_dirs
from src.dataset import create_dataloaders
from src.model import GenreCNN
from src.train import train_model

set_seed(42)
setup_logging()
ensure_dirs()

train_loader, val_loader, test_loader = create_dataloaders()
model = GenreCNN()
history = train_model(model, train_loader, val_loader)
```

## 📈 Performance Targets

| Metric | Target | Model |
|--------|--------|-------|
| Segment Accuracy (3s) | ≥ 82% | CNN |
| Track Accuracy (30s, majority-vote) | ≥ 87% | CNN |
| Baseline Accuracy | 65-72% | Random Forest |
| Inference Latency | < 500ms | CNN on CPU |

---

*Built with ❤️ using PyTorch, FastAPI, and Next.js*
