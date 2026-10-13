import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from web.backend.database import get_active_alerts, create_alert

# --- Configuration ---
ALERT_MIN_RISK_LEVELS = ['HIGH', 'CRITICAL']

# Configuration per risk level
ALERT_CONFIG = {
    'HIGH': {
        'radius_km': 1.0,
        'duration_minutes': 60,
    },
    'CRITICAL': {
        'radius_km': 2.0,
        'duration_minutes': 120,
    }
}

# Distance within which we don't create duplicate alerts (in km)
DEDUPLICATION_RADIUS_KM = 0.5

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def process_incident_for_alerts(incident: Dict[str, Any], incident_id: int) -> Optional[Dict[str, Any]]:
    """
    Process an incident and create an alert if it meets the risk threshold.
    Returns the created alert dict, or None if no alert was created.
    """
    risk_level = incident.get('severity', '').upper()
    
    # 1. Check Threshold
    if risk_level not in ALERT_MIN_RISK_LEVELS:
        return None
        
    lat = incident.get('latitude')
    lng = incident.get('longitude')
    
    # 2. Check Valid Location
    if lat is None or lng is None:
        return None

    # 3. Deduplication Check
    active_alerts = get_active_alerts()
    for alert in active_alerts:
        alert_lat = alert.get('latitude')
        alert_lng = alert.get('longitude')
        if alert_lat is not None and alert_lng is not None:
            dist = haversine_distance(lat, lng, alert_lat, alert_lng)
            if dist <= DEDUPLICATION_RADIUS_KM and alert.get('risk_level') == risk_level:
                # Existing similar alert nearby
                return None

    # 4. Create Alert
    config = ALERT_CONFIG[risk_level]
    created_at = datetime.now()
    expires_at = created_at + timedelta(minutes=config['duration_minutes'])
    
    conf = incident.get('confidence')
    conf_str = f"{int(conf * 100)}%" if conf else "Unknown"
    
    message = f"{risk_level} RISK elephant activity detected.\nConfidence: {conf_str}.\nAlert radius: {config['radius_km']} km."
    
    alert_data = {
        'incident_id': incident_id,
        'alert_type': risk_level,
        'message': message,
        'latitude': lat,
        'longitude': lng,
        'radius_km': config['radius_km'],
        'status': 'ACTIVE',
        'is_active': 1,
        'risk_level': risk_level,
        'risk_score': incident.get('risk_score'),
        'created_at': created_at.isoformat(),
        'expires_at': expires_at.isoformat()
    }
    
    alert_id = create_alert(alert_data)
    alert_data['id'] = alert_id
    
    return alert_data
