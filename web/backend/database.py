"""
MACONFLIC — Database Layer.

SQLite database for storing wildlife incident reports.
Lightweight, file-based — no external database server needed.
"""

import os
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'maconflic.db')


def get_db_path():
    return DB_PATH


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialise the database schema and seed with sample data."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                species TEXT DEFAULT 'Elephant',
                call_type TEXT,
                confidence REAL,
                audio_filename TEXT,
                reporter_name TEXT DEFAULT 'Anonymous',
                reporter_contact TEXT,
                description TEXT,
                severity TEXT DEFAULT 'medium' CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                status TEXT DEFAULT 'reported' CHECK(status IN ('reported', 'verified', 'resolved', 'false_alarm')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER REFERENCES incidents(id),
                alert_type TEXT DEFAULT 'warning',
                message TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                radius_km REAL DEFAULT 5.0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_incidents_location
                ON incidents(latitude, longitude);
            CREATE INDEX IF NOT EXISTS idx_incidents_created
                ON incidents(created_at);
            CREATE INDEX IF NOT EXISTS idx_incidents_severity
                ON incidents(severity);
        """)

        # Seed with sample data if empty
        count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        if count == 0:
            _seed_sample_data(conn)

    print(f"[Database] Initialised at {DB_PATH}")


def _seed_sample_data(conn):
    """Insert sample incident data for demonstration."""
    sample_incidents = [
        # Kerala / Tamil Nadu / Karnataka border region (real elephant corridor areas)
        (10.9258, 76.6823, 'Elephant', 'Rumble', 0.92, None, 'Forest Watcher Rajan',
         '+91-9876543210', 'Herd of 5 elephants spotted near Mudumalai, low-frequency rumbles detected',
         'medium', 'verified'),
        (11.5530, 76.6500, 'Elephant', 'Trumpet', 0.87, None, 'Farmer Suresh',
         '+91-9876543211', 'Single bull elephant trumpeting near Wayanad farmland at night',
         'high', 'reported'),
        (10.1200, 77.0600, 'Elephant', 'Roar', 0.78, None, 'Ranger Meena',
         '+91-9876543212', 'Aggressive roaring heard near Periyar Tiger Reserve boundary',
         'critical', 'verified'),
        (11.4100, 76.7200, 'Elephant', 'Rumble', 0.95, None, 'Eco-tourism Guide',
         None, 'Peaceful herd communication rumbles in Bandipur corridor',
         'low', 'verified'),
        (10.4500, 77.5200, 'Elephant', 'Trumpet', 0.83, None, 'Village Headman',
         '+91-9876543213', 'Two elephants near Topslip, trumpet calls heard during dawn',
         'medium', 'reported'),
        (10.0500, 77.0800, 'Elephant', 'Roar', 0.71, None, 'Anonymous',
         None, 'Suspected elephant roar near Thekkady plantation area',
         'high', 'reported'),
        (12.0900, 76.1000, 'Elephant', 'Rumble', 0.89, None, 'Forest Dept Officer',
         '+91-9876543214', 'Herd crossing near Nagarhole, multiple rumble vocalisations',
         'medium', 'verified'),
        (10.8500, 76.9500, 'Elephant', 'Trumpet', 0.91, None, 'Tea Estate Worker',
         '+91-9876543215', 'Lone elephant near Valparai, high-pitched trumpet alarm call',
         'high', 'verified'),
    ]

    conn.executemany("""
        INSERT INTO incidents (latitude, longitude, species, call_type, confidence,
            audio_filename, reporter_name, reporter_contact, description, severity, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_incidents)

    # Add sample alerts
    sample_alerts = [
        (3, 'critical', '🔴 CRITICAL: Aggressive elephant roar detected near Periyar boundary. '
         'Avoid the area.', 10.1200, 77.0600, 5.0, 1),
        (2, 'warning', '⚠️ WARNING: Bull elephant spotted near Wayanad farmland. '
         'Exercise caution.', 11.5530, 76.6500, 3.0, 1),
        (8, 'warning', '⚠️ WARNING: Lone elephant near Valparai tea estates. '
         'Keep safe distance.', 10.8500, 76.9500, 2.0, 1),
    ]

    conn.executemany("""
        INSERT INTO alerts (incident_id, alert_type, message, latitude, longitude,
            radius_km, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, sample_alerts)

    print(f"  Seeded {len(sample_incidents)} sample incidents and {len(sample_alerts)} alerts")


# ─── CRUD Operations ───

def create_incident(data: dict) -> int:
    """Create a new incident and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO incidents (latitude, longitude, species, call_type, confidence,
                audio_filename, reporter_name, reporter_contact, description, severity, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['latitude'], data['longitude'],
            data.get('species', 'Elephant'),
            data.get('call_type'),
            data.get('confidence'),
            data.get('audio_filename'),
            data.get('reporter_name', 'Anonymous'),
            data.get('reporter_contact'),
            data.get('description'),
            data.get('severity', 'medium'),
            data.get('status', 'reported'),
        ))
        return cursor.lastrowid


def get_incident(incident_id: int) -> dict:
    """Get a single incident by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_incidents(limit=100, offset=0, severity=None, status=None,
                      call_type=None) -> list:
    """Get all incidents with optional filters."""
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if status:
        query += " AND status = ?"
        params.append(status)
    if call_type:
        query += " AND call_type = ?"
        params.append(call_type)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def update_incident(incident_id: int, data: dict) -> bool:
    """Update an existing incident."""
    allowed_fields = ['latitude', 'longitude', 'species', 'call_type', 'confidence',
                      'description', 'severity', 'status', 'reporter_name']
    updates = []
    values = []

    for field in allowed_fields:
        if field in data:
            updates.append(f"{field} = ?")
            values.append(data[field])

    if not updates:
        return False

    updates.append("updated_at = ?")
    values.append(datetime.now().isoformat())
    values.append(incident_id)

    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE incidents SET {', '.join(updates)} WHERE id = ?",
            values
        )
        return cursor.rowcount > 0


def delete_incident(incident_id: int) -> bool:
    """Delete an incident by ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM incidents WHERE id = ?", (incident_id,)
        )
        return cursor.rowcount > 0


# ─── Heatmap / GeoJSON Data ───

def get_incidents_geojson(call_type=None, severity=None) -> dict:
    """
    Return all incidents as GeoJSON FeatureCollection for map display.
    """
    incidents = get_all_incidents(limit=1000, call_type=call_type, severity=severity)

    features = []
    for inc in incidents:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [inc['longitude'], inc['latitude']]
            },
            "properties": {
                "id": inc['id'],
                "species": inc['species'],
                "call_type": inc['call_type'],
                "confidence": inc['confidence'],
                "severity": inc['severity'],
                "status": inc['status'],
                "description": inc['description'],
                "reporter": inc['reporter_name'],
                "created_at": inc['created_at'],
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


# ─── Statistics ───

def get_stats() -> dict:
    """Get dashboard statistics."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        by_severity = dict(conn.execute(
            "SELECT severity, COUNT(*) FROM incidents GROUP BY severity"
        ).fetchall())
        by_call_type = dict(conn.execute(
            "SELECT call_type, COUNT(*) FROM incidents GROUP BY call_type"
        ).fetchall())
        by_status = dict(conn.execute(
            "SELECT status, COUNT(*) FROM incidents GROUP BY status"
        ).fetchall())
        recent = [dict(r) for r in conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 5"
        ).fetchall()]

    return {
        'total_incidents': total,
        'by_severity': by_severity,
        'by_call_type': by_call_type,
        'by_status': by_status,
        'recent_incidents': recent,
    }


# ─── Alerts ───

def get_active_alerts() -> list:
    """Get all active alerts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE is_active = 1 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def create_alert(data: dict) -> int:
    """Create a new alert."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO alerts (incident_id, alert_type, message, latitude, longitude,
                radius_km, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('incident_id'),
            data.get('alert_type', 'warning'),
            data['message'],
            data.get('latitude'),
            data.get('longitude'),
            data.get('radius_km', 5.0),
            data.get('is_active', 1),
        ))
        return cursor.lastrowid


def dismiss_alert(alert_id: int) -> bool:
    """Dismiss (deactivate) an alert."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE alerts SET is_active = 0 WHERE id = ?", (alert_id,)
        )
        return cursor.rowcount > 0


if __name__ == '__main__':
    init_db()
    print("\nStats:", json.dumps(get_stats(), indent=2, default=str))
