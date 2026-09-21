"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ReportPage() {
  const [form, setForm] = useState({
    latitude: "",
    longitude: "",
    call_type: "",
    severity: "medium",
    reporter_name: "",
    reporter_contact: "",
    description: "",
  });
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [locationLoading, setLocationLoading] = useState(false);

  function getCurrentLocation() {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser");
      return;
    }
    setLocationLoading(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setForm((prev) => ({
          ...prev,
          latitude: position.coords.latitude.toFixed(6),
          longitude: position.coords.longitude.toFixed(6),
        }));
        setLocationLoading(false);
      },
      (error) => {
        console.error("Error getting location:", error);
        alert("Failed to get current location. It may have timed out or permissions are missing.");
        setLocationLoading(false);
      },
      { timeout: 5000, maximumAge: 0 }
    );
  }

  function handleChange(e) {
    setForm({ ...form, [e.target.name]: e.target.value });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = {
        latitude: parseFloat(form.latitude),
        longitude: parseFloat(form.longitude),
        species: "Elephant",
        call_type: form.call_type || null,
        severity: form.severity,
        reporter_name: form.reporter_name || "Anonymous",
        reporter_contact: form.reporter_contact || null,
        description: form.description || null,
      };

      const res = await fetch(`${API_BASE}/api/incidents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to submit report");
      }

      setSubmitted(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (submitted) {
    return (
      <div className="page-container">
        <div className="glass-card animate-fade-in" style={{
          textAlign: "center", padding: "4rem 2rem", maxWidth: "600px", margin: "0 auto",
        }}>
          <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>✅</div>
          <h2 style={{ marginBottom: "0.5rem" }}>Report Submitted Successfully!</h2>
          <p style={{ color: "var(--text-secondary)", marginBottom: "2rem" }}>
            Thank you for reporting this wildlife sighting. Our team will review and verify the incident.
          </p>
          <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
            <button
              className="btn btn-primary"
              onClick={() => { setSubmitted(false); setForm({ latitude: "", longitude: "", call_type: "", severity: "medium", reporter_name: "", reporter_contact: "", description: "" }); }}
            >
              📝 Submit Another
            </button>
            <a href="/map" className="btn btn-secondary">🗺️ View Map</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">📝 Report Wildlife Incident</h1>
        <p className="page-subtitle">
          Submit a new elephant sighting or human-wildlife conflict report
        </p>
      </div>

      <div style={{ maxWidth: "700px", margin: "0 auto" }}>
        {error && (
          <div className="alert-banner critical" style={{ marginBottom: "1.5rem" }}>
            <span>❌</span>
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="glass-card" style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ marginBottom: "1.25rem" }}>📍 Location</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div className="form-group">
                <label className="form-label">Latitude *</label>
                <input
                  className="form-input"
                  type="number"
                  name="latitude"
                  step="any"
                  min="-90"
                  max="90"
                  required
                  placeholder="e.g., 10.9258"
                  value={form.latitude}
                  onChange={handleChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Longitude *</label>
                <input
                  className="form-input"
                  type="number"
                  name="longitude"
                  step="any"
                  min="-180"
                  max="180"
                  required
                  placeholder="e.g., 76.6823"
                  value={form.longitude}
                  onChange={handleChange}
                />
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.5rem" }}>
              <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", margin: 0 }}>
                💡 Tip: Use Google Maps to get coordinates, or use your live location.
              </p>
              <button 
                type="button" 
                className="btn btn-secondary" 
                style={{ padding: "0.3rem 0.6rem", fontSize: "0.8rem" }}
                onClick={getCurrentLocation}
                disabled={locationLoading}
              >
                {locationLoading ? "⏳ Locating..." : "📍 Use Current Location"}
              </button>
            </div>
          </div>

          <div className="glass-card" style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ marginBottom: "1.25rem" }}>🐘 Incident Details</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div className="form-group">
                <label className="form-label">Call Type (if known)</label>
                <select className="form-select" name="call_type" value={form.call_type} onChange={handleChange}>
                  <option value="">Unknown / Not Heard</option>
                  <option value="Roar">🦁 Roar — Aggressive</option>
                  <option value="Rumble">🔊 Rumble — Communication</option>
                  <option value="Trumpet">🎺 Trumpet — Alarm/Excitement</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Severity Level *</label>
                <select className="form-select" name="severity" value={form.severity} onChange={handleChange} required>
                  <option value="low">🟢 Low — Peaceful sighting</option>
                  <option value="medium">🟡 Medium — Caution needed</option>
                  <option value="high">🟠 High — Potential conflict</option>
                  <option value="critical">🔴 Critical — Immediate danger</option>
                </select>
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea
                className="form-textarea"
                name="description"
                rows="3"
                placeholder="Describe what you observed (e.g., number of elephants, behavior, location landmarks)..."
                value={form.description}
                onChange={handleChange}
              />
            </div>
          </div>

          <div className="glass-card" style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ marginBottom: "1.25rem" }}>👤 Reporter Information</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div className="form-group">
                <label className="form-label">Your Name</label>
                <input
                  className="form-input"
                  type="text"
                  name="reporter_name"
                  placeholder="e.g., Rajan Kumar"
                  value={form.reporter_name}
                  onChange={handleChange}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Contact Number</label>
                <input
                  className="form-input"
                  type="tel"
                  name="reporter_contact"
                  placeholder="e.g., +91-9876543210"
                  value={form.reporter_contact}
                  onChange={handleChange}
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: "100%", padding: "0.9rem", fontSize: "1rem" }}
          >
            {loading ? "⏳ Submitting..." : "📤 Submit Incident Report"}
          </button>
        </form>
      </div>
    </div>
  );
}
