# ResoGenre — Project Agent Rules

## Project Identity
- **Name:** ResoGenre — Deep-Learning-Based Audio & Music Genre Classification System
- **Type:** End-to-end ML pipeline with Next.js + FastAPI UI
- **Language:** Python 3.10+ (Backend/ML) + TypeScript/Next.js (Frontend)
- **Dataset:** GTZAN (1000 .wav files, 10 genres, 30s each)

---

## Architecture Overview

```
AIDS_mini/
├── data/                    # Raw & processed dataset (GTZAN)
│   ├── raw/                 # Original 30s .wav files by genre
│   └── processed/           # 3s segmented mel-spectrograms (.pt tensors)
├── src/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset, DataLoader, audio augmentation
│   ├── features.py          # Mel-spectrogram extraction, MFCC, handcrafted features
│   ├── model.py             # CNN nn.Module definition (4-stage 2D CNN)
│   ├── train.py             # Training loop, validation, checkpoint saving
│   ├── evaluate.py          # Metrics computation, confusion matrix, F1 scores
│   ├── baseline.py          # Random Forest / XGBoost baseline model
│   └── utils.py             # Seed setting, device selection, path helpers
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, lifespan, model loading
│   ├── routes.py            # /predict, /waveform, /spectrogram endpoints
│   └── schemas.py           # Pydantic request/response models
├── frontend/                # Next.js application (premium UI)
│   ├── src/
│   │   ├── app/             # App router pages
│   │   ├── components/      # React components
│   │   └── styles/          # CSS modules & global styles
│   ├── public/              # Static assets
│   └── package.json
├── notebooks/
│   └── exploration.ipynb    # EDA, waveform & spectrogram visualizations
├── models/                  # Saved model checkpoints (.pth)
├── outputs/                 # Training logs, plots, confusion matrices
├── config.py                # Hyperparameters & global constants
├── requirements.txt         # Pinned Python dependencies
├── README.md
└── IMPLEMENTATION.md        # Phase-wise development tracker
```

---

## Coding Standards

### General Rules
- All random seeds MUST be pinned to `42` (PyTorch, NumPy, Scikit-learn, Python random).
- Use type hints for all function signatures.
- Every public function must have a docstring (Google-style).
- Keep modules focused: one responsibility per file.

### Audio Processing Rules
- Resample ALL audio to `22,050 Hz` mono.
- Mel-spectrogram parameters: `n_fft=2048`, `hop_length=512`, `n_mels=128`.
- Always convert power spectrograms to dB scale via `librosa.power_to_db`.
- Normalize tensor values to `[0, 1]` range after dB conversion.
- Segment 30s clips into 10 × 3s non-overlapping slices.

### Data Splitting Rules (CRITICAL)
- **Leakage Prevention:** Split at the ORIGINAL 30s song level FIRST, then segment.
- Splits: 80% train / 10% validation / 10% test.
- Slices from the same song MUST NOT appear in different partitions.

### Model Architecture
- 4-stage 2D CNN: `[1→32→64→128→256]` channels.
- Each conv block: `Conv2D → BatchNorm2D → ReLU → Pool → Dropout`.
- Final: `AdaptiveAvgPool2d(1,1) → Flatten → FC(256→128) → FC(128→10) → LogSoftmax`.
- Loss: `CrossEntropyLoss` (PyTorch handles log-softmax internally).
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)` with `ReduceLROnPlateau`.

### Baseline Model
- Random Forest with `n_estimators=200`.
- 42 handcrafted features: 20 MFCC means + variances, spectral centroid, rolloff, flux, ZCR, 12 chroma.
- Expected accuracy: ~65–72%.

### FastAPI Backend Rules
- Use async endpoints where possible.
- Load model once at startup via lifespan context manager.
- CORS must allow the Next.js frontend origin.
- Return predictions as JSON with genre, confidence, and full probability distribution.

### Next.js Frontend Rules
- Use App Router (Next.js 14+).
- Dark mode as default with glassmorphism aesthetic.
- All components must be responsive.
- Use CSS Modules or vanilla CSS — no Tailwind.
- Micro-animations on all interactive elements.

### Performance Targets
- CNN test accuracy ≥ 82% on 3s segments.
- Majority-vote ensemble accuracy ≥ 87% on full 30s tracks.
- Inference latency < 500ms for 30s audio on CPU.

---

## Dependencies

### Python (Backend + ML)
```
torch, torchaudio, torchvision
librosa, soundfile
scikit-learn
numpy, pandas
matplotlib, seaborn
fastapi, uvicorn, python-multipart
```

### Frontend (Next.js)
```
next, react, react-dom
```

---

## File Naming Conventions
- Source code: `snake_case.py` (Python), `PascalCase.tsx` (React components)
- Model checkpoints: `resogenre_cnn_epoch{N}.pth`
- Plots: `{plot_type}_{description}.png`
- Data files: lowercase with underscores
- CSS modules: `ComponentName.module.css`
