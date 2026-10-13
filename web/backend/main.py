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
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import requests
import time

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

    # Background task for expiring alerts
    async def check_expired_alerts():
        from web.backend.database import get_connection
        from web.backend.event_manager import event_manager
        from datetime import datetime
        while True:
            try:
                with get_connection() as conn:
                    rows = conn.execute(
                        "SELECT id FROM alerts WHERE status = 'ACTIVE' AND expires_at < ?",
                        (datetime.now().isoformat(),)
                    ).fetchall()
                    if rows:
                        for row in rows:
                            alert_id = row['id']
                            conn.execute(
                                "UPDATE alerts SET status = 'EXPIRED', is_active = 0 WHERE id = ?",
                                (alert_id,)
                            )
                            event_manager.publish_event('alert_expired', {'id': alert_id, 'status': 'EXPIRED'})
            except Exception as e:
                print(f"Error checking expired alerts: {e}")
            await asyncio.sleep(10)
            
    asyncio.create_task(check_expired_alerts())


# ─── Pydantic Models ───

class IncidentCreate(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    species: str = "Elephant"
    call_type: Optional[str] = None
    confidence: Optional[float] = None
    audio_filename: Optional[str] = None
    reporter_name: str = "Anonymous"
    reporter_contact: Optional[str] = None
    description: Optional[str] = None
    severity: str = "medium"
    status: str = "reported"
    source: str = "MANUAL_REPORT"


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
async def predict_audio(
    file: UploadFile = File(...), 
    gradcam: bool = Query(True),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    source: str = Form("AI_AUDIO"),
    reporter_name: str = Form("Web Analyzer (Upload)")
):
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

        incident_created = False
        incident_id = None

        if result['predicted_class'] != 'Non_Elephant':
            from web.backend.risk_engine import calculate_risk
            
            incident_data = {
                'latitude': latitude,
                'longitude': longitude,
                'species': 'Elephant',
                'call_type': result['predicted_class'],
                'confidence': result['confidence'],
                'source': source,
                'reporter_name': reporter_name,
                'description': 'Auto-detected by backend AI prediction.',
                'status': 'reported'
            }
            
            risk_info = calculate_risk(incident_data)
            incident_data['risk_score'] = risk_info['risk_score']
            incident_data['risk_reason'] = risk_info['risk_reason']
            incident_data['severity'] = risk_info['risk_level'].lower()
            
            incident_id = create_incident(incident_data)
            incident_created = True
            
            # Phase 4: Alert Engine
            from web.backend.alert_engine import process_incident_for_alerts
            alert_info = process_incident_for_alerts(incident_data, incident_id)
            
            result['risk_info'] = risk_info
            result['alert_info'] = alert_info

        result['incident_created'] = incident_created
        if incident_created:
            result['incident_id'] = incident_id
            
            # Publish real-time events
            from web.backend.event_manager import event_manager
            from web.backend.database import get_incident
            
            inc_record = get_incident(incident_id)
            if inc_record:
                event_manager.publish_event('incident_created', inc_record)
                
            if result.get('alert_info'):
                event_manager.publish_event('alert_created', result['alert_info'])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─── Incidents CRUD ───

@app.post("/api/incidents", status_code=201)
async def api_create_incident(incident: IncidentCreate):
    """Create a new wildlife incident report."""
    from web.backend.risk_engine import calculate_risk
    
    incident_data = incident.model_dump()
    risk_info = calculate_risk(incident_data)
    
    incident_data['risk_score'] = risk_info['risk_score']
    incident_data['risk_reason'] = risk_info['risk_reason']
    incident_data['severity'] = risk_info['risk_level'].lower()
    
    incident_id = create_incident(incident_data)
    
    # Phase 4: Alert Engine
    from web.backend.alert_engine import process_incident_for_alerts
    alert_info = process_incident_for_alerts(incident_data, incident_id)
    
    # Phase 5: Publish real-time events
    from web.backend.event_manager import event_manager
    from web.backend.database import get_incident
    
    inc_record = get_incident(incident_id)
    if inc_record:
        event_manager.publish_event('incident_created', inc_record)
        
    if alert_info:
        event_manager.publish_event('alert_created', alert_info)
    
    return {
        "id": incident_id,
        "message": "Incident created successfully",
        "risk": risk_info,
        "alert": alert_info
    }


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

@app.get("/api/events/stream")
async def sse_endpoint(request: Request):
    """Server-Sent Events endpoint for real-time updates."""
    from web.backend.event_manager import event_manager
    async def event_generator():
        try:
            async for event in event_manager.subscribe():
                if await request.is_disconnected():
                    break
                yield event
        except asyncio.CancelledError:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    })

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
        
    from web.backend.event_manager import event_manager
    event_manager.publish_event('alert_resolved', {'id': alert_id, 'status': 'RESOLVED'})
    
    return {"message": "Alert dismissed successfully"}


# ─── Sentinel-2 Remote Sensing ───
_sentinel_cache = {
    "data": None,
    "timestamp": 0
}

@app.get("/api/sentinel/search")
async def api_sentinel_search():
    """Query Microsoft Planetary Computer STAC for recent Sentinel-2 imagery."""
    global _sentinel_cache
    if _sentinel_cache["data"] and (time.time() - _sentinel_cache["timestamp"] < 3600):
        return _sentinel_cache["data"]

    bbox = [76.680, 11.080, 76.780, 11.150]
    cloud_thresholds = [20, 30, 40, 60, 100]
    
    try:
        for threshold in cloud_thresholds:
            search_body = {
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "sortby": [{"field": "datetime", "direction": "desc"}],
                "query": {
                    "eo:cloud_cover": {"lt": threshold}
                },
                "limit": 1
            }
            
            response = requests.post(
                "https://planetarycomputer.microsoft.com/api/stac/v1/search",
                json=search_body,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("features") and len(data["features"]) > 0:
                    feature = data["features"][0]
                    result = {
                        "success": True,
                        "source": "Microsoft Planetary Computer",
                        "collection": "sentinel-2-l2a",
                        "item_id": feature["id"],
                        "acquisition_date": feature["properties"]["datetime"],
                        "cloud_cover": feature["properties"]["eo:cloud_cover"],
                        "bbox": feature["bbox"]
                    }
                    _sentinel_cache["data"] = result
                    _sentinel_cache["timestamp"] = time.time()
                    return result
                    
        return {
            "success": False,
            "message": "No suitable Sentinel-2 image found for the study area"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error querying Planetary Computer: {str(e)}"
        }

from fastapi import Response

@app.get("/api/sentinel/tiles/{layer}/{z}/{x}/{y}.png")
async def get_sentinel_tile(layer: str, z: int, x: int, y: int):
    """Proxy STAC API tiles to bypass CORS and SAS token issues."""
    global _sentinel_cache
    if not _sentinel_cache["data"]:
        await api_sentinel_search()
    
    if not _sentinel_cache["data"] or not _sentinel_cache["data"].get("success"):
        raise HTTPException(status_code=404, detail="Sentinel-2 data unavailable")
        
    item_id = _sentinel_cache["data"]["item_id"]
    collection = _sentinel_cache["data"]["collection"]
    
    base_url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{z}/{x}/{y}?collection={collection}&item={item_id}"
    
    if layer == "rgb":
        url = f"{base_url}&assets=visual"
    elif layer == "false-color":
        url = f"{base_url}&assets=B08&assets=B04&assets=B03&color_formula=gamma+r+3.2+gamma+g+3.2+gamma+b+3.2"
    elif layer == "ndvi":
        url = f"{base_url}&expression=(B08-B04)/(B08%2BB04)&colormap_name=rdylgn"
    else:
        raise HTTPException(status_code=400, detail="Invalid layer")
        
    import os
    import tempfile
    _tile_cache_dir = os.path.join(tempfile.gettempdir(), "maconflic_tile_cache")
    os.makedirs(_tile_cache_dir, exist_ok=True)
    
    cache_key = f"{item_id}_{layer}_{z}_{x}_{y}.png"
    cache_file = os.path.join(_tile_cache_dir, cache_key)
    
    headers = {"Cache-Control": "public, max-age=86400"}
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return Response(content=f.read(), media_type="image/png", headers=headers)
            
    try:
        response = requests.get(url, timeout=10.0)
        if response.status_code == 200:
            with open(cache_file, "wb") as f:
                f.write(response.content)
            return Response(content=response.content, media_type="image/png", headers=headers)
        else:
            raise HTTPException(status_code=response.status_code, detail="Tile generation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Run with uvicorn ───
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("web.backend.main:app", host="0.0.0.0", port=8000, reload=True)
