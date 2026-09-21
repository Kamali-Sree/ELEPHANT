import time
import random
import requests

API_URL = "http://127.0.0.1:8000/api/incidents"

CALL_TYPES = ["Roar", "Rumble", "Trumpet"]
SEVERITIES = ["low", "medium", "high", "critical"]

# Base center (Elephant Corridor)
BASE_LAT = 10.9
BASE_LNG = 76.8

def generate_random_incident():
    call = random.choice(CALL_TYPES)
    
    # Assign severity weights
    if call == "Roar":
        severity = random.choices(SEVERITIES, weights=[10, 30, 40, 20])[0]
    elif call == "Trumpet":
        severity = random.choices(SEVERITIES, weights=[5, 20, 50, 25])[0]
    else:
        severity = random.choices(SEVERITIES, weights=[50, 30, 15, 5])[0]

    lat = BASE_LAT + random.uniform(-0.4, 0.4)
    lng = BASE_LNG + random.uniform(-0.4, 0.4)
    conf = random.uniform(0.65, 0.99)

    return {
        "latitude": round(lat, 5),
        "longitude": round(lng, 5),
        "species": "Elephant",
        "call_type": call,
        "severity": severity,
        "confidence": round(conf, 3),
        "reporter_name": f"Acoustic Sensor #{random.randint(100, 999)}",
        "description": "Auto-detected by remote acoustic sensor."
    }

print("=======================================")
print("[MACONFLIC] Live Sensor Simulator Started")
print("=======================================")
print("Sending dynamic live data to the API every 8-15 seconds...\n")

while True:
    try:
        payload = generate_random_incident()
        res = requests.post(API_URL, json=payload)
        
        if res.status_code == 201:
            print(f"[+] SUCCESS: Logged {payload['call_type']} at {payload['latitude']}, {payload['longitude']}")
        else:
            print(f"[-] FAILED: {res.text}")
    except Exception as e:
        print(f"[-] ERROR: Could not connect to API: {e}")

    # Wait random time between 8 to 15 seconds
    delay = random.uniform(8, 15)
    time.sleep(delay)
