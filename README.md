# 🐘 MACONFLIC: Elephant Acoustic Communication Signals to Mitigate Human–Animal Conflict Using AI & ML

**Project:** Analysis of Elephant Acoustic Communication Signals to Mitigate Human–Animal Conflict Using AI & ML Techniques  
**Objective:** To develop a real-time, AI-based acoustic analysis and early warning platform for detecting elephant vocalizations, plotting live GIS data, and broadcasting alerts to mitigate human-elephant conflict.

---

## 1. Project Overview

MACONFLIC (Mitigating Animal Conflict) is an end-to-end Early Warning System. The core of the platform is a Deep Learning model (Convolutional Neural Network with Attention Mechanisms) that listens to acoustic recordings from forest sensors. It detects and classifies elephant vocalizations into specific call types: **Roar, Rumble, and Trumpet**. 

When a call is detected, it is instantly logged into a cloud database, plotted on a Live GIS Map, and evaluated for severity to broadcast early warnings to nearby villages.

### Key Features
1. **AI Audio Analyzer:** Upload audio recordings and receive instant classification using our advanced CNN, complete with Grad-CAM heatmaps showing exactly what frequencies the AI focused on.
2. **Live GIS Map:** Real-time visualization of all elephant incidents, plotting exact GPS coordinates and conflict severity (Low, Medium, High, Critical).
3. **Automated IoT Sensor Simulation:** Real-time data pipeline where simulated remote acoustic sensors automatically push detected elephant calls to the live system.
4. **Early Warning Alerts:** A broadcast hub for forest rangers to push active danger zones to local communities.
5. **Incident Reporting:** A crowdsourcing form for field rangers and villagers to manually log live sightings via their phone's GPS.

---

## 2. System Architecture

The project has evolved into a modern, production-grade Web Application with three core layers:

### A. AI / Machine Learning Layer (PyTorch)
- **Model Architecture:** We upgraded to an **SE-CNN (Squeeze-and-Excitation CNN)**. This is a 4-block Convolutional Neural Network with an SE-Attention mechanism. The attention block allows the AI to focus on important audio frequencies (like low rumbles) while ignoring background forest noise.
- **Input:** 128-band Log-Mel Spectrograms (16,000 Hz, 6.0s clips).
- **Output:** 4 Classes (Roar, Rumble, Trumpet, Non-Elephant).
- **Explainability:** We integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) which generates visual heatmaps of the spectrograms, allowing researchers to see *why* the AI made its decision.

### B. Backend Server Layer (FastAPI & SQLite)
- **Framework:** Python FastAPI.
- **Purpose:** Exposes RESTful APIs for the frontend to interact with the database, AI model, Risk Engine, and Alert Engine.
- **Endpoints:** 
  - `/api/predict`: Handles audio uploads, runs the PyTorch model, and returns predictions & Grad-CAM images.
  - `/api/incidents`: CRUD operations for wildlife incident data (Latitude, Longitude, Call Type, Severity).
  - `/api/alerts`: Manages active early-warning broadcasts based on geographical risk zones.
  - `/api/events/stream`: Server-Sent Events (SSE) endpoint for real-time dashboard synchronization.

### C. Frontend Web Platform (Next.js & React)
- **Framework:** Next.js (React) with dynamic client-side routing.
- **UI Design:** A modern, responsive, "glassmorphism" interface with a forest-inspired dark theme.
- **Mapping:** Integrated with **Leaflet.js** and OpenStreetMap for interactive, seamless GIS plotting.
- **Real-Time Feel:** Uses Next.js `<Link>` routing for instantaneous navigation between the Dashboard, Map, and Analyzer without page reloads.

---

## 3. Dynamic Real-Time Data Pipeline

To demonstrate how the system works in a real-world scenario with physical microphones placed in the forest, we implemented a **Live IoT Sensor Simulator**:
- A background Python daemon (`scripts/sensor_simulator.py`) runs continuously.
- Every 8 to 15 seconds, it simulates a remote sensor picking up an elephant call.
- It dynamically generates GPS coordinates near known elephant corridors and POSTs the data to the FastAPI backend.
- The web platform (Dashboard and GIS Map) updates in real-time, showing the incident count rising and new markers appearing on the map without manual intervention.

## 4. Full System Architecture & Data Flow

The MACONFLIC pipeline integrates multiple engines to move from raw audio to automated community alerts natively and in real-time.

```text
┌──────────────────────────────────────────┐
│              MACONFLIC                   │
│                                          │
│ Audio / IoT / Manual                     │
│          ↓                               │
│     AI Detection                         │
│          ↓                               │
│      Incident                            │
│          ↓                               │
│     Risk Engine                          │
│          ↓                               │
│    Alert Engine                          │
│          ↓                               │
│     SQLite Database                      │
│          ↓                               │
│       SSE Events                         │
│          ↓                               │
│      Live GIS Map                        │
│          ↓                               │
│  Human / Community Warning               │
└──────────────────────────────────────────┘
```

### Module Specifications
* **AI Detection:** Uses a custom SE-CNN (Squeeze-and-Excitation CNN) trained to classify 4 distinct classes (Roar, Rumble, Trumpet, Non-Elephant). The model operates at an 82.5% test accuracy.
* **Risk Engine:** Evaluates incidents based on location and confidence to generate a Risk Score. *(Note: This score is currently a configurable engineering rule designed for platform demonstration, not a scientifically validated ecological probability.)*
* **Alert Engine:** Automatically evaluates the Risk Level. If an incident reaches `HIGH` or `CRITICAL`, it instantly creates a deduplicated geographical Alert Zone (e.g. 1.0 km radius for HIGH, 2.0 km for CRITICAL).
* **SSE (Real-Time Communication):** Employs an in-memory event manager to push updates instantly to connected browsers. *(Note: The current pub/sub system is in-memory and intended for this single-backend prototype. A production multi-node deployment would require an external broker like Redis.)*

---

## 5. How to Run the Platform

### Step 1: Start the Backend (FastAPI)
```bash
# From the project root folder
cd web/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Start the Frontend (Next.js)
```bash
# In a new terminal
cd web/frontend
npm run dev
```
*The web platform will be available at `http://localhost:3000`.*

### Step 3: Start the Live Sensor Simulator (Optional for Demo)
```bash
# In a new terminal
python scripts/sensor_simulator.py
```
*This will begin pushing live, simulated elephant detections to the map and dashboard every 10 seconds.*

---
*Developed for the mitigation of Human-Elephant Conflict.*
