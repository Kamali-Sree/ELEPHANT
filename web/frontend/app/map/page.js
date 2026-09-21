"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Dynamically import the map component to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import("./MapComponent"), {
  ssr: false,
  loading: () => (
    <div style={{
      height: "600px", background: "var(--bg-secondary)", borderRadius: "var(--radius-lg)",
      display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-secondary)"
    }}>
      Loading map...
    </div>
  ),
});

export default function MapPage() {
  const [geojson, setGeojson] = useState(null);
  const [filters, setFilters] = useState({ call_type: "", severity: "" });
  const [loading, setLoading] = useState(true);
  const [userLocation, setUserLocation] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);

  function locateMe() {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser");
      return;
    }
    setLocationLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLocation([pos.coords.latitude, pos.coords.longitude]);
        setLocationLoading(false);
      },
      (err) => {
        console.error("Location error", err);
        alert("Failed to get location. Your device might not support location services or it timed out.");
        setLocationLoading(false);
      },
      { timeout: 5000, maximumAge: 0 }
    );
  }

  useEffect(() => {
    fetchData();
  }, [filters]);

  async function fetchData() {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filters.call_type) params.set("call_type", filters.call_type);
      if (filters.severity) params.set("severity", filters.severity);

      const res = await fetch(`${API_BASE}/api/incidents/heatmap?${params}`);
      if (res.ok) setGeojson(await res.json());
    } catch (err) {
      console.error("Failed to fetch map data:", err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">🗺️ GIS Incident Map</h1>
        <p className="page-subtitle">
          Interactive map showing elephant vocalization incidents and conflict zones
        </p>
      </div>

      {/* Filters */}
      <div className="glass-card" style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "end" }}>
          <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: "150px" }}>
            <label className="form-label">Call Type</label>
            <select
              className="form-select"
              value={filters.call_type}
              onChange={(e) => setFilters({ ...filters, call_type: e.target.value })}
            >
              <option value="">All Types</option>
              <option value="Roar">🦁 Roar</option>
              <option value="Rumble">🔊 Rumble</option>
              <option value="Trumpet">🎺 Trumpet</option>
            </select>
          </div>
          <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: "150px" }}>
            <label className="form-label">Severity</label>
            <select
              className="form-select"
              value={filters.severity}
              onChange={(e) => setFilters({ ...filters, severity: e.target.value })}
            >
              <option value="">All Severities</option>
              <option value="low">🟢 Low</option>
              <option value="medium">🟡 Medium</option>
              <option value="high">🟠 High</option>
              <option value="critical">🔴 Critical</option>
            </select>
          </div>
          <div>
            <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginRight: "1rem" }}>
              {geojson?.features?.length || 0} incidents shown
            </span>
            <button 
              className="btn btn-secondary" 
              style={{ padding: "0.4rem 0.8rem", fontSize: "0.85rem" }}
              onClick={locateMe}
              disabled={locationLoading}
            >
              {locationLoading ? "⏳ Locating..." : "📍 Go to My Location"}
            </button>
          </div>
        </div>
      </div>

      {/* Map */}
      <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
        <MapComponent geojson={geojson} userLocation={userLocation} />
      </div>

      {/* Legend */}
      <div className="glass-card" style={{ marginTop: "1.5rem" }}>
        <h3 style={{ marginBottom: "0.75rem" }}>🎨 Legend</h3>
        <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
          <LegendItem color="#22c55e" label="Low Severity" />
          <LegendItem color="#f59e0b" label="Medium Severity" />
          <LegendItem color="#f97316" label="High Severity" />
          <LegendItem color="#ef4444" label="Critical Severity" />
        </div>
      </div>
    </div>
  );
}

function LegendItem({ color, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
      <div style={{
        width: 14, height: 14, borderRadius: "50%",
        background: color, boxShadow: `0 0 8px ${color}50`,
      }} />
      <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{label}</span>
    </div>
  );
}
