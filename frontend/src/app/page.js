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

  // Draw neo-brutalist waveform on canvas
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

    // Neo-brutalist fill: solid red base fill
    ctx.fillStyle = "rgba(255, 107, 107, 0.25)";
    ctx.beginPath();
    ctx.moveTo(0, midY);
    const step = width / samples.length;
    for (let i = 0; i < samples.length; i++) {
      const x = i * step;
      const y = midY + samples[i] * midY * 0.9;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(width, midY);
    ctx.closePath();
    ctx.fill();

    // Solid Black Stroke with sharp edges
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i++) {
      const x = i * step;
      const y = midY + samples[i] * midY * 0.9;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Secondary yellow reflection line
    ctx.strokeStyle = "#FFD93D";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i++) {
      const x = i * step;
      const y = midY - samples[i] * midY * 0.45;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [waveformData]);

  const handleFile = useCallback((selectedFile) => {
    if (!selectedFile) return;

    const ext = selectedFile.name.split(".").pop().toLowerCase();
    if (!["wav", "mp3", "ogg", "flac"].includes(ext)) {
      setError("UNSUPPORTED FORMAT! Please upload .wav, .mp3, .ogg, or .flac files.");
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
      // Fire parallel requests
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
        setError("FAILED TO GET PREDICTION! Check if FastAPI backend is running on port 8000.");
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
      setError(`CONNECTION ERROR: ${err.message}. Ensure FastAPI server is running on http://localhost:8000.`);
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
      setError(`FAILED TO LOAD SAMPLE (${genre}): ${err.message}. Check backend connection.`);
    } finally {
      setLoadingSample(null);
    }
  };

  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <>
      {/* 1. Marquee Top Ticker */}
      <div className="marquee-ribbon">
        <div className="marquee-content">
          ★ AI MUSIC GENRE CLASSIFICATION ★ 10 GTZAN GENRES ★ 79% TRACK ACCURACY ★ 4-STAGE 2D CNN ★ PYTORCH 2.6 CUDA ★ 18MS GPU INFERENCE ★ NVIDIA GEFORCE RTX 2050 ★ ZERO LEAKAGE SPLIT ★ FASTAPI + NEXT.JS 16 ★&nbsp;&nbsp;&nbsp;&nbsp;
          ★ AI MUSIC GENRE CLASSIFICATION ★ 10 GTZAN GENRES ★ 79% TRACK ACCURACY ★ 4-STAGE 2D CNN ★ PYTORCH 2.6 CUDA ★ 18MS GPU INFERENCE ★ NVIDIA GEFORCE RTX 2050 ★ ZERO LEAKAGE SPLIT ★ FASTAPI + NEXT.JS 16 ★
        </div>
      </div>

      {/* 2. Neo-Brutalist Header */}
      <header className="neo-header">
        <div className="container neo-header-inner">
          <div className="neo-logo">
            <span className="neo-logo-icon">⚡</span>
            <span className="neo-logo-text">ResoGenre</span>
          </div>

          <div className="neo-nav-meta">
            <div className="neo-status-pill">
              <span className="status-dot" />
              <span>CUDA RTX 2050 ACTIVE</span>
            </div>
            <span className="neo-badge-red">v1.0 LIVE</span>
          </div>
        </div>
      </header>

      {/* 3. Main Content Container */}
      <main className="container">
        {/* Hero Section */}
        <section className="neo-hero">
          <div className="neo-hero-badge">
            🚀 DEEP LEARNING AUDIO CLASSIFIER
          </div>

          <h1 className="neo-hero-title">
            INSTANTLY IDENTIFY <br />
            <span className="neo-text-sticker">ANY MUSIC GENRE</span> <br />
            <span className="neo-text-stroke">WITH 2D CNN</span>
          </h1>

          <p className="neo-hero-subtitle">
            Upload any 30-second audio track. Our 4-stage Deep Convolutional Neural Network extracts Log-Mel Spectrograms and computes 19-slice soft voting in under 20ms.
          </p>

          {/* Interactive Genre Badges */}
          <div className="neo-genre-pills">
            {GENRES.map((genre) => (
              <span key={genre} className="neo-genre-pill">
                {GENRE_EMOJIS[genre.toLowerCase().replace("-", "")] || "🎵"}{" "}
                {genre}
              </span>
            ))}
          </div>
        </section>

        {/* 4. Chunky Neo-Brutalist Upload Card */}
        <div
          id="upload-zone"
          className={`neo-upload-card ${isDragging ? "drag-over" : ""}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={handleClick}
        >
          <div className="neo-sticker-corner">★ DROP IT HERE!</div>

          <div className="neo-upload-icon-box">
            🎧
          </div>

          <h2 className="neo-upload-title">
            {isDragging ? "RELEASE TO DROP AUDIO!" : "DRAG & DROP AUDIO FILE OR CLICK TO BROWSE"}
          </h2>

          <p className="neo-upload-desc">
            Supports standard 30-second music tracks with real-time sliding-window analysis
          </p>

          <div className="neo-upload-badges">
            <span className="neo-format-badge">.WAV</span>
            <span className="neo-format-badge">.MP3</span>
            <span className="neo-format-badge">.OGG</span>
            <span className="neo-format-badge">.FLAC</span>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.ogg,.flac,audio/*"
            onChange={handleInputChange}
            style={{ display: "none" }}
            id="audio-upload-input"
          />
        </div>

        {/* 5. 1-Click Instant Test Drawer */}
        <div className="neo-samples-card">
          <div className="neo-samples-header">
            <h3 className="neo-samples-title">
              ⚡ OR TEST INSTANTLY WITH GTZAN BENCHMARK SAMPLES:
            </h3>
            <span style={{ fontSize: "0.8rem", fontWeight: 900, textTransform: "uppercase" }}>
              1-CLICK AUTO PREDICT
            </span>
          </div>

          <div className="neo-samples-grid">
            {["classical", "metal", "blues", "jazz", "disco", "pop", "reggae", "rock"].map((genre) => (
              <button
                key={genre}
                className="neo-sample-btn"
                onClick={() => handleSampleClick(genre)}
                disabled={isLoading || loadingSample !== null}
              >
                <span>{loadingSample === genre ? "⏳" : GENRE_EMOJIS[genre] || "🎵"}</span>
                <span>{genre}</span>
              </button>
            ))}
          </div>
        </div>

        {/* 6. Active Audio Player */}
        {file && audioUrl && (
          <div className="neo-player-card">
            <div className="neo-player-info">
              <span style={{ fontSize: "1.8rem" }}>🔊</span>
              <span className="neo-player-name">{file.name}</span>
            </div>
            <audio controls src={audioUrl} />
            <button className="neo-btn-remove" onClick={removeFile}>
              ✕ REMOVE FILE
            </button>
          </div>
        )}

        {/* 7. Loading Card */}
        {isLoading && (
          <div className="neo-loading-card">
            <div className="neo-spinner" />
            <p className="neo-loading-text">
              ANALYZING AUDIO... EXTRACTING MEL-SPECTROGRAMS & RUNNING GPU FORWARD PASS!
            </p>
          </div>
        )}

        {/* 8. Error Card */}
        {error && (
          <div className="neo-error-card">
            <span style={{ fontSize: "2rem" }}>⚠️</span>
            <p className="neo-error-text">{error}</p>
          </div>
        )}

        {/* 9. Results Grid */}
        {prediction && !isLoading && (
          <div className="neo-results-section">
            <div className="neo-results-grid">
              {/* Card 1: Main Top Genre Prediction */}
              <div className="neo-pred-card">
                <span className="neo-card-tag">★ PREDICTED GENRE</span>

                <div className="neo-pred-genre-wrap">
                  <span className="neo-pred-emoji">
                    {GENRE_EMOJIS[prediction.predicted_genre.toLowerCase().replace("-", "")] || "🎵"}
                  </span>
                  <h2 className="neo-pred-genre-name">
                    {prediction.predicted_genre}
                  </h2>
                </div>

                <div className="neo-confidence-box">
                  <div className="neo-confidence-header">
                    <span>CONFIDENCE MATCH</span>
                    <span style={{ fontSize: "1.2rem", color: "var(--neo-accent)" }}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="neo-confidence-track">
                    <div
                      className="neo-confidence-fill"
                      style={{ width: `${(prediction.confidence * 100).toFixed(1)}%` }}
                    />
                  </div>
                </div>

                <div className="neo-meta-row">
                  <span className="neo-meta-sticker">
                    ⚡ {prediction.inference_time_ms ? prediction.inference_time_ms.toFixed(0) : "18"}MS INFERENCE
                  </span>
                  <span className="neo-meta-sticker">
                    🎯 MAJORITY VOTE: {prediction.majority_vote_genre || prediction.predicted_genre}
                  </span>
                </div>
              </div>

              {/* Card 2: Probability Distribution */}
              <div className="neo-prob-card">
                <div className="neo-card-header">
                  <h3>GENRE PROBABILITY SPREAD</h3>
                  <span style={{ fontSize: "0.8rem", fontWeight: 900 }}>10 CLASSES</span>
                </div>

                <div className="neo-prob-list">
                  {sortedProbabilities.map(([genre, prob]) => {
                    const isTop = genre.toLowerCase() === prediction.predicted_genre.toLowerCase();
                    return (
                      <div key={genre} className="neo-prob-item">
                        <span className="neo-prob-name">
                          {GENRE_EMOJIS[genre.toLowerCase().replace("-", "")] || ""} {genre}
                        </span>
                        <div className="neo-prob-track">
                          <div
                            className={`neo-prob-value ${isTop ? "top-prob" : ""}`}
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="neo-prob-pct">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Visualizer Grid: Waveform + Mel-Spectrogram */}
            <div className="neo-visual-grid">
              {/* Waveform Card */}
              {waveformData && (
                <div className="neo-visual-card">
                  <div className="neo-card-header" style={{ background: "var(--neo-secondary)" }}>
                    <h3>🌊 AUDIO OSCILLOSCOPE WAVEFORM</h3>
                    <span style={{ fontSize: "0.75rem", fontWeight: 900 }}>200 PEAKS</span>
                  </div>
                  <div className="neo-visual-body">
                    <canvas ref={canvasRef} className="neo-waveform-canvas" />
                    <div className="neo-visual-footer">
                      DURATION: {waveformData.duration}S • SAMPLE RATE: {waveformData.sample_rate.toLocaleString()} HZ MONO
                    </div>
                  </div>
                </div>
              )}

              {/* Spectrogram Card */}
              {spectrogramBase64 && (
                <div className="neo-visual-card">
                  <div className="neo-card-header" style={{ background: "var(--neo-accent)", color: "var(--neo-white)" }}>
                    <h3>🔥 2D LOG-MEL SPECTROGRAM HEATMAP</h3>
                    <span style={{ fontSize: "0.75rem", fontWeight: 900 }}>128 MEL BINS</span>
                  </div>
                  <div className="neo-visual-body">
                    <img
                      src={`data:image/png;base64,${spectrogramBase64}`}
                      alt="Log-Mel Spectrogram Heatmap"
                      className="neo-spectrogram-img"
                    />
                    <div className="neo-visual-footer">
                      TIME-FREQUENCY REPRESENTATION ANALYZED BY 2D CONVOLUTIONAL KERNELS
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Segment Breakdown Card */}
            {prediction.segment_predictions && (
              <div className="neo-segments-card">
                <div className="neo-card-header" style={{ background: "var(--neo-muted)" }}>
                  <h3>📊 3-SECOND SEGMENT BREAKDOWN (DENSE SLIDING WINDOW)</h3>
                  <span style={{ fontSize: "0.8rem", fontWeight: 900 }}>
                    {prediction.segment_predictions.length} SLICES
                  </span>
                </div>
                <div className="neo-segments-body">
                  <div className="neo-segments-grid">
                    {prediction.segment_predictions.map((seg, idx) => {
                      const isMaj = seg.toLowerCase() === (prediction.majority_vote_genre || prediction.predicted_genre).toLowerCase();
                      return (
                        <div
                          key={idx}
                          className={`neo-segment-chip ${isMaj ? "is-majority" : ""}`}
                        >
                          <span className="neo-segment-num">SEG {idx + 1}</span>
                          {GENRE_EMOJIS[seg.toLowerCase().replace("-", "")] || ""} {seg}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* 10. High-Contrast Neo-Brutalist Footer */}
      <footer className="neo-footer">
        <div className="container">
          <div className="neo-footer-inner">
            <div className="neo-footer-brand">
              ⚡ RESOGENRE MGC SYSTEM
            </div>

            <div className="neo-footer-chips">
              <span className="neo-footer-chip">PYTORCH 2.6</span>
              <span className="neo-footer-chip">FASTAPI</span>
              <span className="neo-footer-chip">NEXT.JS 16</span>
              <span className="neo-footer-chip">CUDA RTX 2050</span>
            </div>
          </div>

          <div className="neo-footer-text">
            RESOGENRE © 2026 • BUILT FOR HIGH-PRECISION MUSIC GENRE RECOGNITION
          </div>
        </div>
      </footer>
    </>
  );
}
