# Audio Processing Rules — ResoGenre

## Sampling & Format
- Target sample rate: `22,050 Hz` (always resample if different).
- Force mono: downmix multi-channel to single channel.
- Supported input formats: `.wav`, `.mp3`, `.ogg`.
- Internal processing format: PyTorch float32 tensors.

## Mel-Spectrogram Pipeline
1. Load audio with `librosa.load(path, sr=22050, mono=True)`.
2. Compute mel-spectrogram: `librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)`.
3. Convert to dB: `librosa.power_to_db(S, ref=np.max)`.
4. Normalize to `[0, 1]` using min-max normalization per spectrogram.
5. Convert to PyTorch tensor: `torch.FloatTensor(S_norm).unsqueeze(0)` → shape `[1, 128, T]`.

## Segmentation
- Split 30s clips into 10 × 3.0s segments (non-overlapping).
- Each 3s segment at 22050 Hz = `66,150 samples`.
- Expected mel-spectrogram shape per segment: `[1, 128, 130]` (approximately).
- Pad shorter segments with zeros to maintain consistent shape.

## Handcrafted Features (Baseline)
Extract per segment:
- 20 MFCCs → compute mean and variance → 40 features
- Spectral centroid (mean)
- Spectral rolloff (mean)
- Zero-crossing rate (mean)
- 12 Chroma features (mean)
- **Total: ~55 features** (or a curated subset of 42 as per PRD)

## Data Leakage Prevention (CRITICAL)
- ALWAYS split at the original 30s track level before segmenting.
- Train/Val/Test = 80/10/10 at the song level.
- Verify no song-level overlap across partitions before training.
