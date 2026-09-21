"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CALL_TYPE_ICONS = {
  Roar: "🦁",
  Rumble: "🔊",
  Trumpet: "🎺",
  Non_Elephant: "🌿",
};

const SEVERITY_COLORS = {
  low: "#22c55e",
  medium: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, alertsRes] = await Promise.all([
          fetch(`${API_BASE}/api/stats`),
          fetch(`${API_BASE}/api/alerts`),
        ]);

        if (statsRes.ok) setStats(await statsRes.json());
        if (alertsRes.ok) {
          const data = await alertsRes.json();
          setAlerts(data.alerts || []);
        }
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="page-container" style={{ textAlign: "center", paddingTop: "4rem" }}>
        <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🐘</div>
        <p style={{ color: "var(--text-secondary)" }}>Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Hero Section */}
      <div className="page-header animate-fade-in">
        <h1 className="page-title">
          🐘 MACONFLIC <span style={{ color: "var(--text-secondary)", fontWeight: 400, fontSize: "1.25rem" }}>Dashboard</span>
        </h1>
        <p className="page-subtitle">
          AI-Powered Elephant Vocalization Detection & Conflict Monitoring Platform
        </p>
      </div>

      {/* Active Alerts */}
      {alerts.length > 0 && (
        <div className="animate-slide-down" style={{ marginBottom: "1.5rem" }}>
          {alerts.slice(0, 3).map((alert) => (
            <div
              key={alert.id}
              className={`alert-banner ${alert.alert_type === "critical" ? "critical" : "warning"}`}
            >
              <span>{alert.alert_type === "critical" ? "🔴" : "⚠️"}</span>
              <span style={{ flex: 1 }}>{alert.message}</span>
              <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>
                {new Date(alert.created_at).toLocaleDateString()}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid-stats animate-fade-in">
        <StatCard
          icon="📋"
          value={stats?.total_incidents || 0}
          label="Total Incidents"
          color="var(--accent-green)"
        />
        <StatCard
          icon="🔴"
          value={stats?.by_severity?.critical || 0}
          label="Critical"
          color="var(--severity-critical)"
        />
        <StatCard
          icon="🟠"
          value={stats?.by_severity?.high || 0}
          label="High Severity"
          color="var(--severity-high)"
        />
        <StatCard
          icon="✅"
          value={stats?.by_status?.verified || 0}
          label="Verified"
          color="var(--accent-green)"
        />
      </div>

      {/* Two-Column Layout */}
      <div className="grid-2" style={{ marginBottom: "2rem" }}>
        {/* Call Type Distribution */}
        <div className="glass-card">
          <h3 style={{ marginBottom: "1.25rem" }}>📊 Call Type Distribution</h3>
          {stats?.by_call_type && Object.entries(stats.by_call_type)
            .filter(([type]) => type && type !== "null")
            .map(([type, count]) => {
            const total = stats.total_incidents || 1;
            const pct = Math.round((count / total) * 100);
            return (
              <div key={type} style={{ marginBottom: "1.2rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                  <span style={{ fontSize: "0.95rem", fontWeight: 500, display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <span style={{ filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.3))" }}>{CALL_TYPE_ICONS[type] || "🔈"}</span> 
                    {type}
                  </span>
                  <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 600 }}>
                    {count} ({pct}%)
                  </span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Severity Breakdown */}
        <div className="glass-card">
          <h3 style={{ marginBottom: "1.25rem" }}>⚡ Severity Breakdown</h3>
          {stats?.by_severity && Object.entries(stats.by_severity).map(([severity, count]) => {
            const total = stats.total_incidents || 1;
            const pct = Math.round((count / total) * 100);
            const color = SEVERITY_COLORS[severity] || "#6b7280";
            return (
              <div key={severity} style={{ marginBottom: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                  <span className={`badge badge-${severity}`}>{severity}</span>
                  <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                    {count} incidents
                  </span>
                </div>
                <div className="progress-bar">
                  <div style={{
                    height: "100%", borderRadius: "999px",
                    width: `${pct}%`, background: color,
                    transition: "width 0.6s ease",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Incidents Table */}
      <div className="glass-card animate-fade-in">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
          <h3>🕐 Recent Incidents</h3>
          <Link href="/map" className="btn btn-secondary" style={{ fontSize: "0.8rem", padding: "0.4rem 0.8rem" }}>
            View Map →
          </Link>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Type</th>
                <th>Location</th>
                <th>Severity</th>
                <th>Confidence</th>
                <th>Reporter</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {stats?.recent_incidents?.map((inc) => (
                <tr key={inc.id}>
                  <td>
                    {CALL_TYPE_ICONS[inc.call_type] || "🔈"} {inc.call_type}
                  </td>
                  <td style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                    {inc.latitude?.toFixed(4)}, {inc.longitude?.toFixed(4)}
                  </td>
                  <td>
                    <span className={`badge badge-${inc.severity}`}>{inc.severity}</span>
                  </td>
                  <td style={{ color: "var(--accent-green)" }}>
                    {inc.confidence ? `${(inc.confidence * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td style={{ color: "var(--text-secondary)" }}>{inc.reporter_name}</td>
                  <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                    {inc.created_at ? new Date(inc.created_at).toLocaleDateString() : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid-3" style={{ marginTop: "2rem" }}>
        <Link href="/analyze" className="glass-card" style={{ textDecoration: "none", textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>🔬</div>
          <h3>Audio Analyzer</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.4rem" }}>
            Upload audio to classify elephant vocalizations with AI
          </p>
        </Link>
        <Link href="/report" className="glass-card" style={{ textDecoration: "none", textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>📝</div>
          <h3>Report Incident</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.4rem" }}>
            Submit a new wildlife sighting or conflict report
          </p>
        </Link>
        <Link href="/map" className="glass-card" style={{ textDecoration: "none", textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>🗺️</div>
          <h3>Live Map</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.4rem" }}>
            View incident heatmap and conflict zones on GIS
          </p>
        </Link>
      </div>
    </div>
  );
}

function StatCard({ icon, value, label, color }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = value;
    if (end === 0) return;
    const duration = 1000;
    const stepTime = Math.max(Math.floor(duration / end), 30);
    const timer = setInterval(() => {
      start += 1;
      setDisplayValue(start);
      if (start >= end) clearInterval(timer);
    }, stepTime);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <div className="stat-card">
      <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>{icon}</div>
      <div className="stat-value" style={{
        background: `linear-gradient(135deg, ${color}, ${color}cc)`,
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
        backgroundClip: "text",
      }}>
        {displayValue}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
