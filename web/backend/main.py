"""
MACONFLIC — FastAPI Backend Server.

Provides REST API endpoints for:
    - /api/predict       — Audio file classification (upload → prediction)
    - /api/incidents     — CRUD for wildlife incident reports
    - /api/incidents/heatmap — GeoJSON data for map display
    - /api/stats         — Dashboard statistics
    - /api/alerts        — Alert management

Run:
    cd elephant_vocalization_detection
    uvicorn web.backend.main:app --reload --port 8000
"""

import os
import sys
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.backend.database import (
    init_db, create_incident, get_incident, get_all_incidents,
    update_incident, delete_incident, get_incidents_geojson,
    get_stats, get_active_alerts, create_alert, dismiss_alert
)
from web.backend.ml_service import ElephantMLService

# ─── App Setup ───
app = FastAPI(
    title="MACONFLIC API",
    description="AI-Based Elephant Vocalization Detection & Conflict Monitoring Platform",
    version="2.0.0",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Initialise Services ───
ml_service = None


@app.on_event("startup")
async def startup():
    global ml_service
    print("[MACONFLIC] Starting backend server...")

    # Init database
    init_db()

    # Init ML service
    try:
        ml_service = ElephantMLService()
        print("[MACONFLIC] ML service loaded successfully")
    except FileNotFoundError as e:
        print(f"[MACONFLIC] [WARNING] ML service unavailable: {e}")
        ml_service = None


# ─── Pydantic Models ───

class IncidentCreate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    species: str = "Elephant"
    call_type: Optional[str] = None
    confidence: Optional[float] = None
    audio_filename: Optional[str] = None
    reporter_name: str = "Anonymous"
    reporter_contact: Optional[str] = None
    description: Optional[str] = None
    severity: str = "medium"
    status: str = "reported"


class IncidentUpdate(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    species: Optional[str] = None
    call_type: Optional[str] = None
    confidence: Optional[float] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
    reporter_name: Optional[str] = None


class AlertCreate(BaseModel):
    incident_id: Optional[int] = None
    alert_type: str = "warning"
    message: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: float = 5.0
    is_active: int = 1


# ─── API Endpoints ───

# Health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_service": ml_service is not None,
        "model_path": ml_service.model_path if ml_service else None,
    }


# ─── ML Prediction ───

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...), gradcam: bool = Query(True)):
    """
    Upload an audio file and get classification result.

    Args:
        file: Audio file (WAV, MP3, FLAC)
        gradcam: Whether to include Grad-CAM heatmap (default: True)

    Returns:
        Prediction result with class, confidence, probabilities,
        acoustic features, and optionally Grad-CAM image.
    """
    if ml_service is None:
        raise HTTPException(status_code=503, detail="ML service not available. Train the model first.")

    # Validate file type
    allowed = {'.wav', '.mp3', '.flac', '.ogg'}
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"upload{ext}")

    try:
        with open(temp_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        if gradcam:
            result = ml_service.predict_with_gradcam(temp_path)
        else:
            result = ml_service.predict(temp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─── Incidents CRUD ───

@app.post("/api/incidents", status_code=201)
async def api_create_incident(incident: IncidentCreate):
    """Create a new wildlife incident report."""
    incident_id = create_incident(incident.model_dump())
    return {"id": incident_id, "message": "Incident created successfully"}


@app.get("/api/incidents")
async def api_get_incidents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    severity: Optional[str] = None,
    status: Optional[str] = None,
    call_type: Optional[str] = None,
):
    """Get all incidents with optional filters."""
    incidents = get_all_incidents(
        limit=limit, offset=offset,
        severity=severity, status=status, call_type=call_type
    )
    return {"incidents": incidents, "count": len(incidents)}


@app.get("/api/incidents/heatmap")
async def api_get_heatmap(
    call_type: Optional[str] = None,
    severity: Optional[str] = None,
):
    """Get incidents as GeoJSON for map display."""
    return get_incidents_geojson(call_type=call_type, severity=severity)


@app.get("/api/incidents/{incident_id}")
async def api_get_incident(incident_id: int):
    """Get a single incident by ID."""
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@app.put("/api/incidents/{incident_id}")
async def api_update_incident(incident_id: int, data: IncidentUpdate):
    """Update an existing incident."""
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    if not update_incident(incident_id, update_data):
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"message": "Incident updated successfully"}


@app.delete("/api/incidents/{incident_id}")
async def api_delete_incident(incident_id: int):
    """Delete an incident."""
    if not delete_incident(incident_id):
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"message": "Incident deleted successfully"}


# ─── Statistics ───

@app.get("/api/stats")
async def api_get_stats():
    """Get dashboard statistics."""
    return get_stats()


# ─── Alerts ───

@app.get("/api/alerts")
async def api_get_alerts():
    """Get all active alerts."""
    alerts = get_active_alerts()
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/api/alerts", status_code=201)
async def api_create_alert(alert: AlertCreate):
    """Create a new alert."""
    alert_id = create_alert(alert.model_dump())
    return {"id": alert_id, "message": "Alert created successfully"}


@app.put("/api/alerts/{alert_id}/dismiss")
async def api_dismiss_alert(alert_id: int):
    """Dismiss an alert."""
    if not dismiss_alert(alert_id):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert dismissed successfully"}


# ─── Run with uvicorn ───
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("web.backend.main:app", host="0.0.0.0", port=8000, reload=True)
