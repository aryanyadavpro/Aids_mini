"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const GENRES = [
  "Blues", "Classical", "Country", "Disco", "Hip-Hop",
  "Jazz", "Metal", "Pop", "Reggae", "Rock",
];

const GENRE_EMOJIS = {
  blues: "🎷",
  classical: "🎻",
  country: "🤠",
  disco: "🪩",
  hiphop: "🎤",
  jazz: "🎺",
  metal: "🤘",
  pop: "🎵",
  reggae: "🌴",
  rock: "🎸",
};

export default function Home() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [waveformData, setWaveformData] = useState(null);
  const [spectrogramBase64, setSpectrogramBase64] = useState(null);
  const [error, setError] = useState(null);
  const [loadingSample, setLoadingSample] = useState(null);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  // Draw waveform on canvas
  useEffect(() => {
    if (!waveformData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const samples = waveformData.samples;
    const midY = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, "rgba(139, 92, 246, 0.8)");
    gradient.addColorStop(0.5, "rgba(6, 182, 212, 0.8)");
    gradient.addColorStop(1, "rgba(139, 92, 246, 0.8)");

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const step = width / samples.length;
    for (let i = 0; i < samples.length; i++) {
      const x = i * step;
      const y = midY + samples[i] * midY * 0.9;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Mirror (subtle reflection)
    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = gradient;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i++) {
      const x = i * step;
      const y = midY - samples[i] * midY * 0.4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }, [waveformData]);

  const handleFile = useCallback((selectedFile) => {
    if (!selectedFile) return;

    const ext = selectedFile.name.split(".").pop().toLowerCase();
    if (!["wav", "mp3", "ogg"].includes(ext)) {
      setError("Unsupported format. Please upload .wav, .mp3, or .ogg files.");
      return;
    }

    setFile(selectedFile);
    setAudioUrl(URL.createObjectURL(selectedFile));
    setPrediction(null);
    setWaveformData(null);
    setSpectrogramBase64(null);
    setError(null);

    // Auto-analyze
    analyzeAudio(selectedFile);
  }, []);

  const analyzeAudio = async (audioFile) => {
    setIsLoading(true);
    setError(null);

    try {
      // Fire all three requests in parallel
      const formData1 = new FormData();
      formData1.append("file", audioFile);

      const formData2 = new FormData();
      formData2.append("file", audioFile);

      const formData3 = new FormData();
      formData3.append("file", audioFile);

      const [predRes, waveRes, specRes] = await Promise.allSettled([
        fetch(`${API_BASE}/predict`, { method: "POST", body: formData1 }),
        fetch(`${API_BASE}/waveform`, { method: "POST", body: formData2 }),
        fetch(`${API_BASE}/spectrogram`, { method: "POST", body: formData3 }),
      ]);

      if (predRes.status === "fulfilled" && predRes.value.ok) {
        const data = await predRes.value.json();
        setPrediction(data);
      } else {
        setError("Failed to get prediction. Is the backend running?");
      }

      if (waveRes.status === "fulfilled" && waveRes.value.ok) {
        const data = await waveRes.value.json();
        setWaveformData(data);
      }

      if (specRes.status === "fulfilled" && specRes.value.ok) {
        const data = await specRes.value.json();
        setSpectrogramBase64(data.image_base64);
      }
    } catch (err) {
      setError(`Connection error: ${err.message}. Make sure the FastAPI backend is running on port 8000.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    handleFile(droppedFile);
  }, [handleFile]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleClick = () => fileInputRef.current?.click();

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const removeFile = () => {
    setFile(null);
    setAudioUrl(null);
    setPrediction(null);
    setWaveformData(null);
    setSpectrogramBase64(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSampleClick = async (genre) => {
    try {
      setLoadingSample(genre);
      setError(null);
      const res = await fetch(`${API_BASE}/sample/${genre}`);
      if (!res.ok) throw new Error(`Could not load sample audio for ${genre}`);
      const blob = await res.blob();
      const sampleFile = new File([blob], `${genre}_sample.wav`, { type: "audio/wav" });
      handleFile(sampleFile);
    } catch (err) {
      setError(`Failed to load ${genre} sample: ${err.message}. Is the backend running?`);
    } finally {
      setLoadingSample(null);
    }
  };

  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities)
        .sort((a, b) => b[1] - a[1])
    : [];

  return (
    <>
      {/* Header */}
      <header className="header">
        <div className="container header-inner">
          <div className="logo">
            <span className="logo-icon">🎵</span>
            <span className="logo-text">ResoGenre</span>
          </div>
          <span className="nav-badge">AI-Powered</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content container">
        {/* Hero */}
        <section className="hero">
          <h1 className="hero-title">
            Identify Any Music Genre<br />with Deep Learning
          </h1>
          <p className="hero-subtitle">
            Upload an audio file and our 4-stage CNN instantly classifies it
            across 10 genres with real-time confidence scoring and spectral analysis.
          </p>
          <div className="genre-tags">
            {GENRES.map((genre) => (
              <span key={genre} className="genre-tag">
                {GENRE_EMOJIS[genre.toLowerCase().replace("-", "")] || "🎵"}{" "}
                {genre}
              </span>
            ))}
          </div>
        </section>

        {/* Upload Zone */}
        <div
          id="upload-zone"
          className={`upload-zone ${isDragging ? "drag-over" : ""}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={handleClick}
        >
          <span className="upload-icon">🎧</span>
          <h2 className="upload-title">
            {isDragging ? "Drop your audio file here!" : "Drop audio file or click to browse"}
          </h2>
          <p className="upload-subtitle">
            Supports music clips up to 30 seconds
          </p>
          <div className="upload-formats">
            <span className="format-badge">.wav</span>
            <span className="format-badge">.mp3</span>
            <span className="format-badge">.ogg</span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.ogg,audio/*"
            onChange={handleInputChange}
            style={{ display: "none" }}
            id="audio-upload-input"
          />
        </div>

        {/* Sample Tracks for 1-Click Testing */}
        <div className="samples-wrapper animate-in">
          <p className="samples-label">⚡ Or test instantly with sample audio from GTZAN dataset</p>
          <div className="samples-row">
            {["classical", "metal", "blues", "jazz", "disco", "pop", "reggae", "rock"].map((genre) => (
              <button
                key={genre}
                className="sample-chip"
                onClick={() => handleSampleClick(genre)}
                disabled={isLoading || loadingSample !== null}
              >
                <span className="chip-icon">
                  {loadingSample === genre ? "⏳" : GENRE_EMOJIS[genre] || "🎵"}
                </span>
                <span style={{ textTransform: "capitalize" }}>{genre}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Audio Player */}
        {file && audioUrl && (
          <div className="audio-player animate-in">
            <span>🎵</span>
            <span className="audio-filename">{file.name}</span>
            <audio controls src={audioUrl} />
            <button className="remove-btn" onClick={removeFile}>
              ✕ Remove
            </button>
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="glass-card loading-overlay animate-in" style={{ marginTop: 40 }}>
            <div className="spinner" />
            <p className="loading-text">Analyzing audio... Extracting mel-spectrograms & classifying</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="glass-card error-card animate-in" style={{ marginTop: 40 }}>
            <span className="error-icon">⚠️</span>
            <p className="error-message">{error}</p>
          </div>
        )}

        {/* Results */}
        {prediction && !isLoading && (
          <div className="results-grid animate-in">
            {/* Prediction Card */}
            <div className="glass-card prediction-card animate-in animate-in-delay-1">
              <p className="prediction-label">Predicted Genre</p>
              <p className="prediction-genre">
                {GENRE_EMOJIS[prediction.predicted_genre.toLowerCase().replace("-", "")] || "🎵"}{" "}
                {prediction.predicted_genre}
              </p>
              <div className="confidence-meter">
                <div className="confidence-bar-bg">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${(prediction.confidence * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
              <p className="confidence-value">
                {(prediction.confidence * 100).toFixed(1)}%
              </p>
              <p className="inference-time">
                ⚡ {prediction.inference_time_ms.toFixed(0)}ms inference
              </p>
            </div>

            {/* Probability Bars */}
            <div className="glass-card prob-card animate-in animate-in-delay-2">
              <h3>Genre Probabilities</h3>
              {sortedProbabilities.map(([genre, prob]) => (
                <div key={genre} className="prob-bar-item">
                  <span className="prob-genre-name">
                    {GENRE_EMOJIS[genre.toLowerCase().replace("-", "")] || ""} {genre}
                  </span>
                  <div className="prob-bar-track">
                    <div
                      className={`prob-bar-value ${
                        genre === prediction.predicted_genre ? "top-genre" : "other-genre"
                      }`}
                      style={{ width: `${(prob * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="prob-percentage">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>

            {/* Waveform */}
            {waveformData && (
              <div className="glass-card visualization-card animate-in animate-in-delay-3">
                <h3>🌊 Audio Waveform</h3>
                <canvas ref={canvasRef} className="waveform-canvas" />
                <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8 }}>
                  Duration: {waveformData.duration}s • Sample Rate: {waveformData.sample_rate.toLocaleString()} Hz
                </p>
              </div>
            )}

            {/* Spectrogram */}
            {spectrogramBase64 && (
              <div className="glass-card visualization-card animate-in animate-in-delay-4">
                <h3>🔥 Mel-Spectrogram</h3>
                <img
                  src={`data:image/png;base64,${spectrogramBase64}`}
                  alt="Mel-Spectrogram"
                  className="spectrogram-img"
                />
              </div>
            )}

            {/* Segment Predictions */}
            {prediction.segment_predictions && (
              <div className="glass-card segments-card results-full animate-in">
                <h3>📊 Per-Segment Predictions (3s slices)</h3>
                <div className="segments-grid">
                  {prediction.segment_predictions.map((seg, idx) => (
                    <div
                      key={idx}
                      className={`segment-chip ${
                        seg === prediction.majority_vote_genre ? "is-majority" : ""
                      }`}
                    >
                      <span className="segment-label">Seg {idx + 1}</span>
                      {GENRE_EMOJIS[seg.toLowerCase().replace("-", "")] || ""} {seg}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p className="footer-text">
            Built with ❤️ using{" "}
            <span className="footer-link">PyTorch</span> •{" "}
            <span className="footer-link">FastAPI</span> •{" "}
            <span className="footer-link">Next.js</span>
            {" "}— ResoGenre © 2026
          </p>
        </div>
      </footer>
    </>
  );
}
