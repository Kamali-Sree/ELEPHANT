"use client";

import { useState, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newAlert, setNewAlert] = useState({
    alert_type: "warning",
    message: "",
    latitude: "",
    longitude: "",
    radius_km: "5",
  });
  const [locationLoading, setLocationLoading] = useState(false);

  function getCurrentLocation() {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser");
      return;
    }
    setLocationLoading(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setNewAlert((prev) => ({
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

  useEffect(() => {
    fetchAlerts();
  }, []);

  async function fetchAlerts() {
    try {
      const res = await fetch(`${API_BASE}/api/alerts`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts || []);
      }
    } catch (err) {
      console.error("Failed to fetch alerts:", err);
    } finally {
      setLoading(false);
    }
  }

  async function dismissAlert(alertId) {
    try {
      const res = await fetch(`${API_BASE}/api/alerts/${alertId}/dismiss`, {
        method: "PUT",
      });
      if (res.ok) {
        setAlerts(alerts.filter((a) => a.id !== alertId));
      }
    } catch (err) {
      console.error("Failed to dismiss alert:", err);
    }
  }

  async function createAlert(e) {
    e.preventDefault();
    try {
      const payload = {
        alert_type: newAlert.alert_type,
        message: newAlert.message,
        latitude: newAlert.latitude ? parseFloat(newAlert.latitude) : null,
        longitude: newAlert.longitude ? parseFloat(newAlert.longitude) : null,
        radius_km: parseFloat(newAlert.radius_km) || 5.0,
      };

      const res = await fetch(`${API_BASE}/api/alerts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setShowCreate(false);
        setNewAlert({ alert_type: "warning", message: "", latitude: "", longitude: "", radius_km: "5" });
        fetchAlerts();
      }
    } catch (err) {
      console.error("Failed to create alert:", err);
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
          <div>
            <h1 className="page-title">🔔 Alert Center</h1>
            <p className="page-subtitle">
              View active alerts and configure notification zones for early warning
            </p>
          </div>
          <button
            className="btn btn-primary"
            onClick={() => setShowCreate(!showCreate)}
          >
            {showCreate ? "✕ Cancel" : "➕ New Alert"}
          </button>
        </div>
      </div>

      {/* Create Alert Form */}
      {showCreate && (
        <div className="glass-card animate-slide-down" style={{ marginBottom: "1.5rem" }}>
          <h3 style={{ marginBottom: "1rem" }}>🆕 Create New Alert</h3>
          <form onSubmit={createAlert}>
            <div className="form-group">
              <label className="form-label">Alert Type</label>
              <select
                className="form-select"
                value={newAlert.alert_type}
                onChange={(e) => setNewAlert({ ...newAlert, alert_type: e.target.value })}
              >
                <option value="info">ℹ️ Information</option>
                <option value="warning">⚠️ Warning</option>
                <option value="critical">🔴 Critical</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Alert Message *</label>
              <textarea
                className="form-textarea"
                rows="2"
                required
                placeholder="Enter alert message to broadcast..."
                value={newAlert.message}
                onChange={(e) => setNewAlert({ ...newAlert, message: e.target.value })}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1rem" }}>
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
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
              <div className="form-group">
                <label className="form-label">Latitude</label>
                <input
                  className="form-input"
                  type="number"
                  step="any"
                  placeholder="Optional"
                  value={newAlert.latitude}
                  onChange={(e) => setNewAlert({ ...newAlert, latitude: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Longitude</label>
                <input
                  className="form-input"
                  type="number"
                  step="any"
                  placeholder="Optional"
                  value={newAlert.longitude}
                  onChange={(e) => setNewAlert({ ...newAlert, longitude: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Radius (km)</label>
                <input
                  className="form-input"
                  type="number"
                  step="0.5"
                  min="0.5"
                  value={newAlert.radius_km}
                  onChange={(e) => setNewAlert({ ...newAlert, radius_km: e.target.value })}
                />
              </div>
            </div>
            <button type="submit" className="btn btn-primary" style={{ marginTop: "0.5rem" }}>
              📢 Broadcast Alert
            </button>
          </form>
        </div>
      )}

      {/* Stats */}
      <div className="grid-stats" style={{ marginBottom: "1.5rem" }}>
        <div className="stat-card">
          <div style={{ fontSize: "1.5rem" }}>🔔</div>
          <div className="stat-value">{alerts.length}</div>
          <div className="stat-label">Active Alerts</div>
        </div>
        <div className="stat-card">
          <div style={{ fontSize: "1.5rem" }}>🔴</div>
          <div className="stat-value" style={{
            background: "linear-gradient(135deg, var(--accent-red), #f97316)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text",
          }}>
            {alerts.filter((a) => a.alert_type === "critical").length}
          </div>
          <div className="stat-label">Critical</div>
        </div>
        <div className="stat-card">
          <div style={{ fontSize: "1.5rem" }}>⚠️</div>
          <div className="stat-value" style={{
            background: "linear-gradient(135deg, var(--accent-amber), #eab308)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text",
          }}>
            {alerts.filter((a) => a.alert_type === "warning").length}
          </div>
          <div className="stat-label">Warnings</div>
        </div>
      </div>

      {/* Alert List */}
      {loading ? (
        <div className="glass-card" style={{ textAlign: "center", padding: "3rem" }}>
          <p style={{ color: "var(--text-secondary)" }}>Loading alerts...</p>
        </div>
      ) : alerts.length === 0 ? (
        <div className="glass-card" style={{ textAlign: "center", padding: "3rem" }}>
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>✅</div>
          <h3 style={{ marginBottom: "0.5rem" }}>No Active Alerts</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            All clear! There are no active alerts at this time.
          </p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="glass-card"
              style={{
                borderLeft: `4px solid ${
                  alert.alert_type === "critical" ? "var(--accent-red)" :
                  alert.alert_type === "warning" ? "var(--accent-amber)" : "var(--accent-blue)"
                }`,
              }}
            >
              <div style={{ display: "flex", alignItems: "start", gap: "1rem" }}>
                <div style={{ fontSize: "1.5rem", flexShrink: 0 }}>
                  {alert.alert_type === "critical" ? "🔴" :
                   alert.alert_type === "warning" ? "⚠️" : "ℹ️"}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "0.5rem" }}>
                    <span className={`badge badge-${
                      alert.alert_type === "critical" ? "critical" :
                      alert.alert_type === "warning" ? "high" : "medium"
                    }`}>
                      {alert.alert_type}
                    </span>
                    <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                      {alert.created_at ? new Date(alert.created_at).toLocaleString() : ""}
                    </span>
                  </div>
                  <p style={{ marginBottom: "0.5rem" }}>{alert.message}</p>
                  {alert.latitude && (
                    <p style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                      📍 {alert.latitude?.toFixed(4)}, {alert.longitude?.toFixed(4)} — {alert.radius_km} km radius
                    </p>
                  )}
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem", flexShrink: 0 }}
                  onClick={() => dismissAlert(alert.id)}
                >
                  Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
