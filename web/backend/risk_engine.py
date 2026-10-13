import math
from typing import Dict, Any

# --- Demo Configuration for Risk Engine ---
# Since no GIS dataset exists yet, we define a mock settlement for testing.
DEMO_VILLAGE_LAT = 10.92
DEMO_VILLAGE_LNG = 76.85

HIGH_RISK_DISTANCE_KM = 2.0
MEDIUM_RISK_DISTANCE_KM = 5.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance in kilometers between two points."""
    R = 6371.0  # Earth radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_risk(incident: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate risk level and score for an incident based on:
    1. Call type
    2. AI Confidence
    3. Distance to nearest settlement
    """
    score = 0.0
    reasons = []

    # 1. Call Type Weight
    call_type = incident.get('call_type', 'Unknown')
    if call_type == 'Roar':
        score += 30.0
        reasons.append("Aggressive Roar detected (+30)")
    elif call_type == 'Trumpet':
        score += 20.0
        reasons.append("Trumpet detected (+20)")
    elif call_type == 'Rumble':
        score += 10.0
        reasons.append("Rumble detected (+10)")
    elif call_type == 'Non_Elephant':
        return {
            "risk_level": "LOW",
            "risk_score": 0,
            "risk_reason": "No elephant threat (Non_Elephant).",
            "distance_km": None
        }

    # 2. AI Confidence
    confidence = incident.get('confidence')
    if confidence is not None:
        conf_bonus = (confidence - 0.5) * 20.0  # e.g., 1.0 -> +10, 0.5 -> 0
        if conf_bonus > 0:
            score += conf_bonus
            reasons.append(f"High AI Confidence ({confidence:.2f}) (+{conf_bonus:.1f})")
    
    # 3. Location (Distance to Settlement)
    lat = incident.get('latitude')
    lng = incident.get('longitude')
    distance_km = None

    if lat is not None and lng is not None:
        distance_km = haversine_distance(lat, lng, DEMO_VILLAGE_LAT, DEMO_VILLAGE_LNG)
        if distance_km <= HIGH_RISK_DISTANCE_KM:
            score += 50.0
            reasons.append(f"Very close to settlement ({distance_km:.2f} km) (+50)")
        elif distance_km <= MEDIUM_RISK_DISTANCE_KM:
            score += 25.0
            reasons.append(f"Near settlement ({distance_km:.2f} km) (+25)")
        else:
            reasons.append(f"Far from settlement ({distance_km:.2f} km) (+0)")
    else:
        reasons.append("Missing GPS location — location risk could not be evaluated.")

    # Clamp score 0-100
    final_score = max(0, min(100, int(score)))

    # Determine Level
    if final_score < 25:
        level = "LOW"
    elif final_score < 50:
        level = "MEDIUM"
    elif final_score < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "risk_level": level,
        "risk_score": final_score,
        "risk_reason": " | ".join(reasons),
        "distance_km": round(distance_km, 2) if distance_km is not None else None
    }
