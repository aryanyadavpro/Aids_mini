# 🎵 ResoGenre — Music Upload & Prediction Lifecycle (Quick Reference)

> **Step-by-Step Technical Flow of how ResoGenre classifies an uploaded audio track.**

---

## 🌊 Complete Lifecycle at a Glance

```
[ User Uploads .mp3 / .wav ]
             │
             ▼
[ 1. Next.js 16 UI ] ──> Sends audio as FormData to POST /predict
             │
             ▼
[ 2. Backend Ingestion ] ──> Loaded via Librosa & Resampled to 22,050 Hz Mono
             │
             ▼
[ 3. Sliding Window ] ──> 30s song divided into 19 overlapping 3s slices (50% overlap)
             │
             ▼
[ 4. Feature Extraction ] ──> 19 Slices converted to Log-Mel Spectrogram Tensors [1, 128, 130]
             │
             ▼
[ 5. Batched GPU Forward Pass ] ──> torch.stack(19 tensors) run through 4-Stage 2D CNN in ~18ms
             │
             ▼
[ 6. Soft & Hard Voting ] ──> Probabilities averaged across all 19 slices ──> Top Genre decided!
             │
             ▼
[ 7. UI Display ] ──> Genre: Jazz (78.66%), Confidence Bars, Oscilloscope Waveform & Spectrogram
```

---

## 📌 Step-by-Step Breakdown

### Step 1: User Upload & Frontend Trigger (`page.js`)
1. User file upload box me apni `.wav`, `.mp3`, `.ogg`, ya `.flac` file drop karta hai (ya 1-Click test button dabata hai).
2. React frontend `analyzeAudio()` function trigger karta hai.
3. Frontend ek sath backend ko **3 parallel requests** bhejta hai via `Promise.allSettled`:
   - `POST /predict`: Deep learning genre classification ke liye.
   - `POST /waveform`: Canvas oscilloscope ke liye 200 amplitude points nikalne ke liye.
   - `POST /spectrogram`: Visual Log-Mel Spectrogram heatmap photo ke liye.

---

### Step 2: Audio Loading aur Resampling (`src/features.py`)
Jab audio backend me aati hai:
1. **Mono Conversion:** Agar gaana stereo (2 channels: Left aur Right) me hai, toh dono channels ko average karke single **Mono** channel banaya jata hai.
2. **22,050 Hz Resampling:** Gaaney ka sample rate chahe 44.1 kHz ho ya 48 kHz, usse standard **22,050 Hz** par resample kiya jata hai.
   > **Nyquist Theorem:** 22,050 Hz se hum 11,025 Hz tak ki frequencies capture kar sakte hain. Music ke instruments (bass, vocals, guitar, cymbals) isi range me hote hain, aur memory 50% bachti hai.
3. **Dynamic Peak Normalization:**
   $$y_{\text{norm}} = \frac{y}{\max(|y|)} \times 0.95$$
   Agar song dheema (quiet) record hua hai, toh uske volume ko scale kiya jata hai taaki quiet instruments (jaise piano chords ya flute) model ko saaf sunai dein.

---

### Step 3: 50% Overlapping Sliding Window (19 Slices Banana)
Model pure 30 second ke gaane ko ek sath nahi dekhta. Kyu?
- Pure 30s ka ek bada spectrogram banane par memory bahut zyada lagti hai aur gaane ke beech ka change miss ho sakta hai.
- **Humara Logic:** Hum gaane ko **3.0 second ke chote tukdon (slices)** me kaatte hain with **1.5 second overlap (hop)**:
  - Slice 1: `0.0s` se `3.0s`
  - Slice 2: `1.5s` se `4.5s`
  - Slice 3: `3.0s` se `6.0s`
  - ...
  - Slice 19: `27.0s` se `30.0s`
- **Total: 19 Slices.** Har slice me $22,050 \times 3 = \mathbf{66,150\text{ audio samples}}$ hote hain.

---

### Step 4: Audio ko "Spectrogram Image" me Badalna (DSP)
CNN direct sound wave nahi padh sakta, usse 2D image chahiye hoti hai. Isliye har 3-second slice par **DSP (Digital Signal Processing)** lagti hai:

1. **STFT (Short-Time Fourier Transform):**
   - $2048$ samples (~93 ms) ka window lekar Fourier Transform karte hain, jo batata hai ki us 93 ms me kaun-kaun se frequencies baje. Hop length $512$ samples (~23 ms) rakha gaya hai.
2. **128 Mel Filterbank:**
   - Insaan ke kaan low frequencies (bass/kick drum) me chota sa badlav bhi pehchan lete hain, lekin high frequencies me unhe farq kam pata chalta hai. 
   - Mel-scale linear Hertz ko insaani hearing scale me convert karta hai:
     $$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$
3. **Decibels (dB) Scale:**
   - Volume ko linear se logarithmic decibels me convert karte hain ($10 \cdot \log_{10}(P)$) aur values ko $[0.0, 1.0]$ me normalize karte hain.
- **Output Tensor Shape:** Har 3s slice ban jati hai ek 2D matrix shape `[1, 128, 130]`:
  - `1`: Single channel (Grayscale image ki tarah).
  - `128`: Frequency rows (Mel pitch bands).
  - `130`: Time columns (130 frames over 3 seconds).

---

### Step 5: GPU Batch Forward Pass (4-Stage 2D CNN)
Ab backend ke paas 19 slices ke 19 tensors hain.
1. In 19 tensors ko ek sath stack karke batch banaya jata hai: **`[19, 1, 128, 130]`**.
2. Is pure batch ko direct **NVIDIA RTX 2050 GPU** ki VRAM memory me bhejte hain.
3. **GenreCNN Model Layers:**
   - **Stage 1 (1 → 32 filters):** Gaane ke basic edges, transients, aur notes detect karta hai.
   - **Stage 2 (32 → 64 filters):** Harmonics aur basic drum beats pehchanta hai.
   - **Stage 3 (64 → 128 filters):** Basslines, tempo rhythm, aur chord transitions pehchanta hai.
   - **Stage 4 (128 → 256 filters):** Heavy genre patterns pehchanta hai (jaise Metal ki distorted electric guitar, Jazz ka ride cymbal swing, Hip-hop ka 808 sub-bass).
   - **Global Average Pooling:** Spatial features ko 256 dimensions me compress karta hai.
   - **Dense Layers:** 256 → 128 → 10 classes ke probabilities me convert karta hai.
- Ye pura forward pass GPU par sirf **15 se 20 milliseconds** me complete ho jata hai!

---

### Step 6: Dual Voting (Soft Voting + Hard Consensus)
Har slice ka ek prediction aata hai. Ab final faisla kaise hota hai?

#### A. Soft Voting (Final Confidence & Genre):
Sabhi 19 slices ke probability distributions ka average nikala jata hai:
$$P_{\text{final}}(\text{genre}) = \frac{P_1 + P_2 + \dots + P_{19}}{19}$$
> **Fayda:** Agar gaane ke beech me 1-second ka koi guitar pause ya khamoshi aayi, toh akele us slice ke galat hone par bhi baaki 18 slices milkar sahi prediction ko top par rakhti hain.

#### B. Hard Voting (Segments Agreement):
Har slice ne kis genre ko vote kiya, wo count hota hai:
> *Example:* 19 me se 19 slices ne bola **Jazz** $\to$ Consensus = **100% (19/19)**.

---

### Step 7: UI me Result Display
FastAPI backend JSON response frontend ko bhejta hai:
```json
{
  "genre": "jazz",
  "confidence": 0.7866,
  "top_3": [
    {"genre": "jazz", "confidence": 0.7866},
    {"genre": "blues", "confidence": 0.0812},
    {"genre": "classical", "confidence": 0.0451}
  ],
  "majority_vote": {
    "genre": "jazz",
    "votes": 19,
    "total_segments": 19,
    "consensus_ratio": 1.0
  }
}
```

Next.js Neo-Brutalist UI is data ko render karta hai:
- 🎺 **Main Genre Card:** Bada "Jazz" icon aur "78.7% Match" with 12px solid black shadow.
- 📊 **Animated Progress Bars:** Sabhi 10 genres ka distribution.
- 🌊 **Live Canvas Waveform:** Gaane ki waves black stroke aur red base fill ke sath animate hoti hain.
- 🌈 **Mel Spectrogram:** Sound ki actual visual frequency heatmap dikhayi deti hai.

---

## 💡 1-Minute Summary (Viva / Presentation me Bolne ke Liye)

> *"Jab user audio upload karta hai, backend usse 22,050 Hz mono me normalize karta hai aur 50% overlapping sliding window se 19 slices (3s each) banata hai. Har slice ko Short-Time Fourier Transform aur 128-Mel filterbank se guzar kar [1, 128, 130] Log-Mel Spectrogram me badla jata hai. Ye 19 tensors RTX 2050 GPU par 4-Stage 2D CNN se sirf 18ms me process hote hain, aur Soft Voting se sabhi slices ka average probability nikal kar final genre aur confidence percentage predict hota hai."*