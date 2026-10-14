# Product Requirements Document (PRD)

## Project Title
**ResoGenre: Deep-Learning-Based Audio & Music Genre Classification System**

---

## 1. Executive Summary & Objective
The objective of this project is to build an end-to-end machine learning system capable of taking an arbitrary audio clip (`.wav` or `.mp3`), computing acoustic spectral representations, and categorizing the track into its corresponding genre with high accuracy. 

The project delivers:
1. A reproducible audio preprocessing pipeline.
2. A dual-model benchmark comparing classical feature-based ML against a deep 2D CNN.
3. Diagnostic visualizations (Mel-spectrograms, multi-class confusion matrix, loss curves).
4. A lightweight browser-based inference application built with Streamlit.
5. A comprehensive slide deck summarizing problem definitions, experiments, and results.

---

## 2. Target Users & Use Cases
* **Academic Reviewers / Evaluators:** Assessing algorithm selection, mathematical understanding, and model performance metrics.
* **Music Streaming Services:** Automated tagging and indexing of user-uploaded tracks into genre taxonomy.
* **Digital DJs & Content Creators:** Quick cataloging and playlist categorization of unlabelled local music files.

---

## 3. Core Functional Requirements

### FR-1: Audio Preprocessing Engine
* **Input:** Audio file formats `.wav`, `.mp3`, `.ogg`.
* **Standardization:**
  - Resample incoming audio to a unified sampling rate $f_s = 22,050\text{ Hz}$.
  - Downmix multi-channel tracks to single-channel mono.
  - Trim or pad clips into uniform 3.0-second intervals.
* **Feature Extraction:**
  - Generate 128-band Mel-Spectrograms ($\text{n\_fft}=2048$, $\text{hop\_length}=512$).
  - Convert to log-amplitude decibel scale (`librosa.power_to_db`).
  - Normalize tensor values to $[0, 1]$ or standardized z-scores.

### FR-2: Modeling & Inference Engine
* **Baseline Model:** Scikit-learn Random Forest trained on 42 handcrafted audio features (MFCCs, spectral centroid, spectral rolloff, chroma).
* **Primary Deep Model:** 4-stage 2D Convolutional Neural Network (CNN) processing 2D Mel-Spectrogram representations.
* **Output:**
  - Top-1 predicted genre label.
  - Softmax probability distribution across all 10 target classes.

### FR-3: Performance & Evaluation Requirements
* **Accuracy:** Target minimum test accuracy $\ge 82\%$ on 3-second segments and $\ge 87\%$ via majority-voting ensemble across 30-second tracks.
* **Inference Latency:** $< 500\text{ ms}$ for a 30-second audio track on modern CPU.
* **Metrics Reported:**
  - Macro and Weighted F1-Score.
  - Precision & Recall per genre.
  - Normalized Multi-Class Confusion Matrix.

### FR-4: User Interface (Streamlit Application)
* **File Upload Component:** Drag-and-drop support for `.wav` and `.mp3` files.
* **Audio Player:** In-browser playback of the uploaded file.
* **Visualization Deck:**
  - Real-time display of the raw audio waveform.
  - Computed Mel-Spectrogram heatmap.
* **Prediction Card:**
  - Highlighted primary predicted genre with confidence percentage.
  - Interactive horizontal bar chart displaying prediction probabilities for all 10 genres.

---

## 4. Non-Functional Requirements
* **Reproducibility:** Pinned `requirements.txt` file and fixed random seeds (`seed=42`) across PyTorch, NumPy, and Scikit-learn.
* **Modularity:** Separation of concerns between:
  - `src/dataset.py`: Audio loading, augmentation, and PyTorch `Dataset` definition.
  - `src/model.py`: PyTorch `nn.Module` definition.
  - `src/train.py`: Training loops, validation passes, and checkpoint saving.
  - `app.py`: Streamlit frontend application.
* **Compute Footprint:** Trainable in under 15 minutes using a single free Google Colab T4 GPU.

---

## 5. Academic Deliverables & Milestone Schedule

| Milestone | Deliverables | Status |
|---|---|---|
| **Phase 1: Setup & Preprocessing** | Dataset acquisition, audio slicing scripts, PyTorch `DataLoader` | Week 1 |
| **Phase 2: Baseline & Model Training** | Random Forest baseline, 2D CNN architecture, training loss/accuracy tracking | Week 2 |
| **Phase 3: Evaluation & Visualization** | Confusion matrix, F1-scores, spectrogram inspection plots | Week 3 |
| **Phase 4: UI & PPT Deck** | Interactive Streamlit demo, finalized presentation slides | Week 4 |

---

## 6. Slide Deck (PPT) Structure
As required by the assignment guidelines, the presentation will follow this 6-slide structure:
1. **Title & Problem Statement:** Academic context, motivation for audio classification, dataset introduction.
2. **Feature Engineering & Signal Processing:** STFT theory, Mel-scale justification, waveform vs. spectrogram contrast.
3. **Model Architectures:** Baseline comparison vs. Proposed 2D CNN architecture diagram.
4. **Experimental Results & Metrics:** Training/validation curves, accuracy/F1 breakdown per genre.
5. **Diagnostic Error Analysis:** Confusion matrix highlights (e.g., distinguishing Rock vs. Metal, Country vs. Folk).
6. **Live Demonstration & Conclusion:** Streamlit application walk-through and future work (Transformers/Audio Spectrogram Transformers).