# Context & Technical Foundation: Music Genre Classification

## 1. Problem Formulation & Theoretical Background

### 1.1 Problem Statement
Music genre classification is a multi-class supervised learning task. Given an arbitrary audio signal $x(t)$, the objective is to predict a categorical label $\hat{y} \in \{C_1, C_2, \dots, C_K\}$, where each $C_k$ represents a standardized musical genre (e.g., *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock*).

### 1.2 Mathematical Formulation of Audio Representation
Raw audio samples are 1D discrete time-series amplitude signals sampled at frequency $f_s$ (commonly $22,050\text{ Hz}$). Raw 1D signals are poorly suited for standard convolutional operations due to phase ambiguity and lack of translation invariance across pitch and time. Therefore, the signal is transformed into the time-frequency domain using the **Short-Time Fourier Transform (STFT)**:

$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$

Where:
- $x[n]$ is the discrete audio signal.
- $w[n]$ is an analysis window (typically a Hann or Hamming window of size $N = 2048$, hop size $H = 512$).
- $m$ is the time index, and $\omega$ is the angular frequency.

### 1.3 Mel-Spectrogram & MFCC Generation
Human pitch perception is logarithmic rather than linear. To simulate human hearing resolution:
1. **Mel Scale Mapping:** The linear frequency $f$ (in Hz) is warped into the Mel scale $m_{mel}$:
   $$m_{mel} = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
2. **Triangular Filterbank:** The power spectrum $|X(m, \omega)|^2$ is filtered across $M$ overlapping triangular Mel filters (typically $M = 128$).
3. **Log Dynamic Range Compression:**
   $$S_{mel}(m, k) = \log\left( |X(m, \omega)|^2 \cdot H_k(\omega) + \epsilon \right)$$
4. **Output Feature Map:** $S_{mel}$ yields a 2D matrix (time frames $\times$ Mel bands) that can be treated identically to a single-channel grayscale image tensor:
   $$\mathbf{X} \in \mathbb{R}^{1 \times F \times T}$$
   where $F = 128$ frequency bands and $T$ is the number of time frames.

---

## 2. Dataset Specification

### Primary Benchmark: GTZAN Dataset
* **Source:** Marsyas Library / Kaggle Audio Datasets.
* **Size:** 1,000 `.wav` files (100 tracks per genre across 10 genres).
* **Duration:** 30 seconds per audio clip at $22,050\text{ Hz}$ mono, 16-bit.
* **Classes ($K=10$):**
  1. Blues
  2. Classical
  3. Country
  4. Disco
  5. Hiphop
  6. Jazz
  7. Metal
  8. Pop
  9. Reggae
  10. Rock

### Data Augmentation & Segment Splitting Strategy
A key failure mode with GTZAN is overfitting due to the modest sample count (1,000 files). 
* **3-Second Segmenting:** Each 30-second audio track is segmented into 10 non-overlapping 3-second slices.
* **Dataset Multiplier:** Expands the training set from $1,000 \to 10,000$ independent examples.
* **Leakage Prevention Rule:** Slices from the same original 30-second track **must not** appear across both train and test partitions. Train/validation/test splitting ($80/10/10$) must occur at the original song level *before* slicing.

---

## 3. Algorithmic Architecture

### 3.1 Baseline: Classical Machine Learning
* **Extracted Features:**
  - 20 MFCC means & variances
  - Spectral Centroid, Rolloff, Flux
  - Zero-Crossing Rate (ZCR)
  - Chroma Features (12 pitch classes)
* **Classifier:** Random Forest ($n_{\text{estimators}}=200$) or XGBoost.
* **Target Metric:** $\sim 65\% - 72\%$ test accuracy.

### 3.2 Deep Learning: Custom 2D Convolutional Neural Network (CNN)
The 2D CNN processes the log-Mel spectrogram tensor $\mathbf{X} \in \mathbb{R}^{B \times 1 \times 128 \times 130}$ (for 3-second segments):

```
Input: [B, 1, 128, 130]
   │
   ├── Conv2D(1 -> 32, kernel=3, padding=1) + BatchNorm2D + ReLU + MaxPool2d(2, 2) + Dropout(0.25)
   │     Output: [B, 32, 64, 65]
   │
   ├── Conv2D(32 -> 64, kernel=3, padding=1) + BatchNorm2D + ReLU + MaxPool2d(2, 2) + Dropout(0.25)
   │     Output: [B, 64, 32, 32]
   │
   ├── Conv2D(64 -> 128, kernel=3, padding=1) + BatchNorm2D + ReLU + MaxPool2d(2, 2) + Dropout(0.3)
   │     Output: [B, 128, 16, 16]
   │
   ├── Conv2D(128 -> 256, kernel=3, padding=1) + BatchNorm2D + ReLU + AdaptiveAvgPool2d((1, 1))
   │     Output: [B, 256, 1, 1]
   │
   ├── Flatten -> Linear(256 -> 128) + ReLU + Dropout(0.5)
   │
   └── Linear(128 -> 10) -> LogSoftmax
```

* **Loss Function:** Categorical Cross-Entropy Loss:
  $$\mathcal{L}_{CE} = -\sum_{k=1}^K y_k \log(\hat{y}_k)$$
* **Optimization:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), with ReduceLROnPlateau.

---

## 4. Tech Stack Breakdown

| Layer | Library / Tool | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Primary development environment |
| **Audio Processing** | `librosa`, `soundfile` | STFT, Mel-Spectrogram extraction, decibel conversion |
| **Deep Learning** | `PyTorch`, `torchvision`, `torchaudio` | Model training, GPU tensors, backprop |
| **Classical ML Baseline**| `scikit-learn` | Random Forest baseline, train/test splitting, confusion matrices |
| **Data Manipulation** | `numpy`, `pandas` | Metadata management and batch preprocessing |
| **Visualization** | `matplotlib`, `seaborn` | Waveform, Mel-spectrogram, loss curves, confusion matrices |
| **Interactive UI** | `streamlit` | Drag-and-drop web dashboard for audio inference |
| **Notebook Runtime** | Google Colab / Kaggle Notebooks | Free T4 GPU acceleration |