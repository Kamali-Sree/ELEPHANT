"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const SatelliteMapComponent = dynamic(() => import("./SatelliteMapComponent"), {
  ssr: false,
  loading: () => (
    <div style={{ height: "700px", background: "#111", display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "12px", border: "1px solid #333" }}>
      Loading satellite map...
    </div>
  ),
});

const API_BASE = "http://localhost:8000";
const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_API_KEY;

export default function SatelliteMapPage() {
  const [geojson, setGeojson] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [activeSentinelLayer, setActiveSentinelLayer] = useState(null);
  
  // Create some simulated sensor coordinates to display (approximate Anakatty region)
  const sensors = [
    { latitude: 11.111, longitude: 76.733, name: "Anakatty Base Sensor" },
    { latitude: 11.130, longitude: 76.710, name: "Forest Edge Sensor A" },
    { latitude: 11.090, longitude: 76.750, name: "Corridor Sensor B" },
  ];

  useEffect(() => {
    fetchInitialData();

    // Set up SSE
    const eventSource = new EventSource(`${API_BASE}/api/events/stream`);

    eventSource.addEventListener("incident_created", (e) => {
      try {
        const incident = JSON.parse(e.data);
        if (incident.latitude === null || incident.longitude === null) return;
        
        const newFeature = {
          type: "Feature",
          geometry: { type: "Point", coordinates: [incident.longitude, incident.latitude] },
          properties: incident
        };

        setGeojson((prev) => {
          if (!prev) return { type: "FeatureCollection", features: [newFeature] };
          if (prev.features.some(f => f.properties.id === newFeature.properties.id)) return prev;
          return { ...prev, features: [newFeature, ...prev.features] };
        });
      } catch (err) {}
    });

    eventSource.addEventListener("alert_created", (e) => {
      try {
        const alert = JSON.parse(e.data);
        setAlerts((prev) => {
          if (prev.some(a => a.id === alert.id)) return prev;
          return [alert, ...prev];
        });
      } catch (err) {}
    });

    const handleAlertRemoval = (e) => {
      try {
        const { id } = JSON.parse(e.data);
        setAlerts((prev) => prev.filter(a => a.id !== id));
      } catch (err) {}
    };

    eventSource.addEventListener("alert_resolved", handleAlertRemoval);
    eventSource.addEventListener("alert_expired", handleAlertRemoval);

    return () => eventSource.close();
  }, []);

  async function fetchInitialData() {
    try {
      const res = await fetch(`${API_BASE}/api/incidents/heatmap`);
      if (res.ok) setGeojson(await res.json());

      const alertsRes = await fetch(`${API_BASE}/api/alerts`);
      if (alertsRes.ok) {
        const alertsData = await alertsRes.json();
        setAlerts(alertsData.alerts || []);
      }
    } catch (err) {
      console.error("Failed to fetch map data", err);
    }
  }

  return (
    <div className="page-container">
      <header className="page-header" style={{ borderBottom: "1px solid #333", paddingBottom: "1rem", marginBottom: "1.5rem" }}>
        <h1 style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>🛰️ SATELLITE RISK MAP</h1>
        <p style={{ color: "var(--text-secondary)" }}>High-resolution geographic visualization for the Anakatty Study Area.</p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: "1.5rem", alignItems: "start" }}>
        {/* Map Container */}
        <div style={{ border: "1px solid #333", borderRadius: "12px", overflow: "hidden", background: "#000" }}>
          <SatelliteMapComponent geojson={geojson} alerts={alerts} sensors={sensors} mapTilerKey={MAPTILER_KEY} activeSentinelLayer={activeSentinelLayer} />
        </div>

        {/* Side Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          
          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>MACONFLIC STUDY AREA</h3>
            <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              <div><strong>Location:</strong> Anakatty, Tamil Nadu</div>
              <div><strong>Active Incidents:</strong> {geojson?.features?.length || 0}</div>
              <div><strong>Active Alerts:</strong> {alerts?.length || 0}</div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem", color: "#16a34a" }}>SENTINEL-2 REMOTE SENSING</h3>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button 
                onClick={() => setActiveSentinelLayer(activeSentinelLayer === 'rgb' ? null : 'rgb')}
                style={{ flex: 1, padding: "0.5rem", borderRadius: "8px", background: activeSentinelLayer === 'rgb' ? "#16a34a" : "#333", color: "#fff", border: "1px solid #22c55e", cursor: "pointer", fontSize: "0.8rem", fontWeight: 600, transition: "0.2s" }}
              >
                RGB
              </button>
              <button 
                onClick={() => setActiveSentinelLayer(activeSentinelLayer === 'false-color' ? null : 'false-color')}
                style={{ flex: 1, padding: "0.5rem", borderRadius: "8px", background: activeSentinelLayer === 'false-color' ? "#16a34a" : "#333", color: "#fff", border: "1px solid #22c55e", cursor: "pointer", fontSize: "0.8rem", fontWeight: 600, transition: "0.2s" }}
              >
                FALSE COLOR
              </button>
              <button 
                onClick={() => setActiveSentinelLayer(activeSentinelLayer === 'ndvi' ? null : 'ndvi')}
                style={{ flex: 1, padding: "0.5rem", borderRadius: "8px", background: activeSentinelLayer === 'ndvi' ? "#16a34a" : "#333", color: "#fff", border: "1px solid #22c55e", cursor: "pointer", fontSize: "0.8rem", fontWeight: 600, transition: "0.2s" }}
              >
                NDVI
              </button>
            </div>
            <p style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
              Click to toggle Sentinel-2 spectral imagery. Deselecting returns to Natural Satellite.
            </p>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>LAYERS</h3>
            <ul style={{ fontSize: "0.9rem", color: "var(--text-secondary)", listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              <li>🛰️ Natural Satellite (MapTiler)</li>
              <li>🗺️ MACONFLIC Danger Spots</li>
              <li>🛖 MACONFLIC Settlements</li>
              <li>🔥 Risk Heatmap</li>
              <li>🚨 Risk Zones (HIGH/CRITICAL)</li>
              <li>📍 Incident Locations</li>
              <li>🎙️ SIMULATED SENSOR LOCATIONS</li>
            </ul>
          </div>

          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <h3 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>RISK LEGEND</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.9rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{ color: "#22c55e" }}>🟢</span> LOW</div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{ color: "#f59e0b" }}>🟡</span> MEDIUM</div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{ color: "#f97316" }}>🟠</span> HIGH</div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{ color: "#ef4444" }}>🔴</span> CRITICAL</div>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}
