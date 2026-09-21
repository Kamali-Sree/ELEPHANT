"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const SEVERITY_COLORS = {
  low: "#22c55e",
  medium: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

const CALL_TYPE_ICONS = {
  Roar: "🦁",
  Rumble: "🔊",
  Trumpet: "🎺",
};

export default function MapComponent({ geojson, userLocation }) {
  // Center on Kerala/Tamil Nadu/Karnataka elephant corridor region
  const center = [10.9, 76.8];
  const zoom = 8;

  function FlyToLocation({ location }) {
    const map = useMap();
    useEffect(() => {
      if (location) {
        map.flyTo(location, 12, { animate: true });
      }
    }, [location, map]);
    return null;
  }

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      style={{ height: "600px", width: "100%", borderRadius: "var(--radius-lg)" }}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <FlyToLocation location={userLocation} />

      {userLocation && (
        <CircleMarker
          center={userLocation}
          radius={12}
          pathOptions={{ color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.8, weight: 3 }}
        >
          <Popup>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: "0.9rem", fontWeight: "bold" }}>
              📍 Your Live Location
            </div>
          </Popup>
        </CircleMarker>
      )}

      {geojson?.features?.map((feature) => {
        const { coordinates } = feature.geometry;
        const props = feature.properties;
        const color = SEVERITY_COLORS[props.severity] || "#6b7280";
        const radius = props.severity === "critical" ? 12 : props.severity === "high" ? 10 : 8;

        return (
          <CircleMarker
            key={props.id}
            center={[coordinates[1], coordinates[0]]}
            radius={radius}
            pathOptions={{
              color: color,
              fillColor: color,
              fillOpacity: 0.6,
              weight: 2,
              opacity: 0.9,
            }}
          >
            <Popup>
              <div style={{
                fontFamily: "Inter, sans-serif",
                fontSize: "0.85rem",
                lineHeight: 1.6,
                minWidth: "200px",
              }}>
                <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.5rem" }}>
                  {CALL_TYPE_ICONS[props.call_type] || "🔈"} {props.call_type || "Unknown"}
                </div>
                <div><strong>Severity:</strong> <span style={{ color }}>{props.severity}</span></div>
                <div><strong>Confidence:</strong> {props.confidence ? `${(props.confidence * 100).toFixed(0)}%` : "—"}</div>
                <div><strong>Status:</strong> {props.status}</div>
                <div><strong>Reporter:</strong> {props.reporter || "Anonymous"}</div>
                {props.description && (
                  <div style={{ marginTop: "0.5rem", color: "#999", fontSize: "0.8rem" }}>
                    {props.description}
                  </div>
                )}
                <div style={{ marginTop: "0.5rem", color: "#666", fontSize: "0.75rem" }}>
                  {props.created_at ? new Date(props.created_at).toLocaleString() : ""}
                </div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
