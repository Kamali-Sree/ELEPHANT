"use client";

import { useState, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CALL_TYPE_ICONS = {
  Roar: "🦁",
  Rumble: "🔊",
  Trumpet: "🎺",
  Non_Elephant: "🌿",
};

const CALL_TYPE_DESCRIPTIONS = {
  Roar: "Aggressive vocalization — often signals a threat or territorial display",
  Rumble: "Low-frequency communication — used for herd coordination and bonding",
  Trumpet: "High-pitched alarm/excitement call — signals danger or excitement",
  Non_Elephant: "Background noise — no elephant vocalization detected",
};

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  function handleDrag(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
      setError(null);
    }
  }

  function handleFileSelect(e) {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  }

  async function handleAnalyze() {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/api/predict?gradcam=true`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);

      // --- DYNAMIC DATA: Automatically log positive detections to DB ---
      if (data.predicted_class && data.predicted_class !== "Non_Elephant") {
        try {
          await fetch(`${API_BASE}/api/incidents`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              latitude: (10.9 + (Math.random() * 0.4 - 0.2)).toFixed(5),
              longitude: (76.8 + (Math.random() * 0.4 - 0.2)).toFixed(5),
              species: "Elephant",
              call_type: data.predicted_class,
              severity: data.predicted_class === "Roar" || data.predicted_class === "Trumpet" ? "high" : "medium",
              confidence: data.confidence,
              reporter_name: "Web Analyzer (Upload)",
              description: `Uploaded audio manually analyzed via Web Interface.`
            })
          });
        } catch (dbErr) {
          console.error("Failed to auto-log to database:", dbErr);
        }
      }
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">🔬 Audio Analyzer</h1>
        <p className="page-subtitle">
          Upload an audio file to classify elephant vocalizations using our trained CNN model with Grad-CAM explainability
        </p>
      </div>

      {/* Upload Zone */}
      <div className="glass-card animate-fade-in" style={{ marginBottom: "1.5rem" }}>
        <div
          className={`upload-zone ${dragActive ? "active" : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg"
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />
          <div className="icon">🎵</div>
          {file ? (
            <div>
              <p style={{ fontWeight: 600, marginBottom: "0.25rem" }}>{file.name}</p>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                {(file.size / 1024).toFixed(1)} KB — Click to change
              </p>
            </div>
          ) : (
            <div>
              <p style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
                Drop audio file here or click to browse
              </p>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                Supports WAV, MP3, FLAC, OGG (max 6 seconds recommended)
              </p>
            </div>
          )}
        </div>

        {file && (
          <div style={{ textAlign: "center", marginTop: "1rem" }}>
            <button
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={loading}
              style={{ minWidth: "200px", fontSize: "1rem", padding: "0.8rem 2rem" }}
            >
              {loading ? "⏳ Analyzing..." : "🔍 Analyze Audio"}
            </button>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="alert-banner critical" style={{ marginBottom: "1.5rem" }}>
          <span>❌</span>
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="animate-slide-down">
          {/* Prediction */}
          <div className="glass-card" style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ marginBottom: "1rem" }}>🏷️ Classification Result</h3>
            <div style={{
              display: "flex", alignItems: "center", gap: "1.5rem",
              padding: "1.5rem", background: "var(--bg-secondary)",
              borderRadius: "var(--radius-md)", marginBottom: "1.25rem",
            }}>
              <div style={{ fontSize: "3.5rem" }}>
                {CALL_TYPE_ICONS[result.predicted_class] || "❓"}
              </div>
              <div style={{ flex: 1 }}>
                <h2 style={{ marginBottom: "0.25rem" }}>{result.predicted_class}</h2>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  {CALL_TYPE_DESCRIPTIONS[result.predicted_class] || ""}
                </p>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{
                  fontSize: "2rem", fontWeight: 800,
                  color: result.confidence > 0.8 ? "var(--accent-green)" :
                         result.confidence > 0.5 ? "var(--accent-amber)" : "var(--accent-red)",
                }}>
                  {(result.confidence * 100).toFixed(1)}%
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>Confidence</div>
              </div>
            </div>

            {/* Class Probabilities */}
            <h3 style={{ marginBottom: "0.75rem", fontSize: "1rem" }}>📊 Class Probabilities</h3>
            {result.probabilities && Object.entries(result.probabilities).map(([cls, prob]) => (
              <div key={cls} style={{ marginBottom: "0.75rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                  <span style={{ fontSize: "0.9rem" }}>
                    {CALL_TYPE_ICONS[cls] || ""} {cls}
                  </span>
                  <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{
                    width: `${prob * 100}%`,
                    background: cls === result.predicted_class
                      ? "linear-gradient(90deg, var(--accent-green), var(--accent-amber))"
                      : "var(--text-muted)",
                  }} />
                </div>
              </div>
            ))}
          </div>

          <div className="grid-2">
            {/* Acoustic Features */}
            {result.acoustic_features && (
              <div className="glass-card">
                <h3 style={{ marginBottom: "1rem" }}>📐 Acoustic Characteristics</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                  <MetricCard label="Duration" value={`${result.acoustic_features.duration} sec`} icon="⏱️" />
                  <MetricCard label="Dominant Freq" value={`${result.acoustic_features.dominant_freq} Hz`} icon="📡" />
                  <MetricCard label="RMS Energy" value={result.acoustic_features.rms_energy?.toFixed(4)} icon="🔋" />
                  <MetricCard label="Zero Crossing" value={result.acoustic_features.zero_crossing_rate?.toFixed(4)} icon="〰️" />
                </div>
              </div>
            )}

            {/* Spectrogram */}
            {result.spectrogram_image && (
              <div className="glass-card">
                <h3 style={{ marginBottom: "1rem" }}>🎨 Mel-Spectrogram</h3>
                <img
                  src={`data:image/png;base64,${result.spectrogram_image}`}
                  alt="Mel-Spectrogram"
                  style={{ width: "100%", borderRadius: "var(--radius-sm)" }}
                />
              </div>
            )}
          </div>

          {/* Grad-CAM */}
          {result.gradcam_image && (
            <div className="glass-card" style={{ marginTop: "1.5rem" }}>
              <h3 style={{ marginBottom: "0.5rem" }}>🔥 Grad-CAM — Model Attention Heatmap</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
                Shows which regions of the spectrogram the CNN model focuses on to make its prediction.
                Red/yellow areas indicate high attention; blue/purple areas indicate low attention.
              </p>
              <img
                src={`data:image/png;base64,${result.gradcam_image}`}
                alt="Grad-CAM Heatmap"
                style={{ width: "100%", borderRadius: "var(--radius-sm)" }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, icon }) {
  return (
    <div style={{
      background: "var(--bg-secondary)", borderRadius: "var(--radius-sm)",
      padding: "0.75rem", textAlign: "center",
    }}>
      <div style={{ fontSize: "1.25rem", marginBottom: "0.25rem" }}>{icon}</div>
      <div style={{ fontWeight: 700, fontSize: "1.1rem", color: "var(--accent-green)" }}>{value}</div>
      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.15rem" }}>{label}</div>
    </div>
  );
}
