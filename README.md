# 🐘 MACONFLIC: Elephant Acoustic Communication Signals to Mitigate Human–Animal Conflict Using AI & ML

**Project:** Analysis of Elephant Acoustic Communication Signals to Mitigate Human–Animal Conflict Using AI & ML Techniques  
**Objective:** An end-to-end early warning and spatial intelligence platform combining deep learning acoustic detection, automated IoT telemetry, organic village polygon GIS mapping, satellite imagery, and real-time risk/alert broadcasting.

---

## 📌 Table of Contents
1. [Platform Overview & Key Capabilities](#1-platform-overview--key-capabilities)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Newly Added Modules & Features](#3-newly-added-modules--features)
4. [Machine Learning & Acoustic Analysis](#4-machine-learning--acoustic-analysis)
5. [GIS, Satellite Map & Organic Polygon Cutouts](#5-gis-satellite-map--organic-polygon-cutouts)
6. [Backend API Endpoints](#6-backend-api-endpoints)
7. [Frontend Application Pages](#7-frontend-application-pages)
8. [Directory Structure](#8-directory-structure)
9. [How to Run (Quickstart)](#9-how-to-run-quickstart)
10. [Analytical Scripts & Pipelines](#10-analytical-scripts--pipelines)
11. [Git Collaboration & Pushing Changes](#11-git-collaboration--pushing-changes)

---

## 1. Platform Overview & Key Capabilities

MACONFLIC (Mitigating Animal Conflict) provides wildlife rangers, conservationists, and vulnerable village communities with real-time early warning capabilities:

1. **AI Audio Analyzer**: Classifies audio recordings into elephant call types (**Roar, Rumble, Trumpet, Non-Elephant**) with **Grad-CAM explainability heatmaps** showing frequency activations.
2. **Interactive Satellite Map (`/map/satellite`)**: High-resolution ESRI satellite imagery layered with true-boundary **organic village polygon cutouts**, elevation contours, and dynamic danger buffer zones.
3. **Standard Live GIS Map (`/map`)**: Real-time Leaflet GIS displaying incidents, conflict heatmaps, severity tiers, and live GPS coordinates.
4. **Autonomous Risk & Alert Engines**: Evaluates incoming detections based on proximity to settlements, historical frequency, and signal confidence to automatically declare and broadcast `HIGH` and `CRITICAL` alert zones.
5. **Real-Time Data Streaming (SSE)**: Server-Sent Events dynamically push new sensor detections to connected web clients without manual page refreshes.
6. **Automated IoT Sensor Simulator**: Generates realistic field telemetry near known elephant movement corridors at configurable intervals.
7. **Village Sighting & Incident Reporting**: Community crowdsourcing portal for rangers and villagers to submit verified sightings.

---

## 2. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES & TELEMETRY                        │
│   Field Microphones / IoT Sensors   │   Ranger Sighting Crowdsourcing  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    AI / ML INFERENCE (PyTorch)                         │
│  Log-Mel Spectrogram Extraction ──► SE-CNN Attention Model (v2)        │
│  Class: Roar / Rumble / Trumpet / Non-Elephant + Grad-CAM Heatmap      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     BACKEND ENGINES (FastAPI)                          │
│  • Event Manager (SSE Pub/Sub)                                         │
│  • Risk Engine (Proximity, Frequency & Confidence Scoring)             │
│  • Alert Engine (Dynamic Danger Buffers: 1.0 km HIGH / 2.0 km CRIT)    │
│  • SQLite Database (Incidents, GeoJSON Layers, Alerts, Logs)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  FRONTEND WEB PLATFORM (Next.js)                       │
│  • Dashboard: Key Metrics, Recent Detections, Sensor Health            │
│  • Live GIS Map: Incident Markers, Severity Tiers, Heatmaps            │
│  • Satellite Map: Organic Settlement Cutouts, Contours, Satellite Tiles│
│  • AI Audio Lab: Drag & Drop WAV Upload, Audio Spectrogram, Grad-CAM   │
│  • Alerts Hub: Ranger Broadcast Center, SMS & Siren Dispatch           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Newly Added Modules & Features

### A. Satellite Map & Organic Polygon Cutouts (`web/frontend/app/map/satellite/`)
- Interactive Next.js component (`SatelliteMapComponent.js`) rendering ESRI World Imagery alongside standard street tiles.
- Vectorized organic village cutouts replacing rough circles or bounding boxes with real settlement geometry for 5 focal forest-fringe villages:
  - **Sembukarai**
  - **Kandivazhi**
  - **Pudur**
  - **Panapalli**
  - **Kodanur**
- Settlement metadata cards displaying estimated population, distance to nearest corridor, risk tier, and recent detections.

### B. Analytical Risk & Alert Engines (`web/backend/`)
- **`alert_engine.py`**: Automated geographical buffer creation, deduplication of overlapping alerts, and severity categorization.
- **`risk_engine.py`**: Proximity-weighted risk score generation combining distance to settlements, vocalization urgency (Roar/Trumpet vs Rumble), and sensor confidence.
- **`event_manager.py`**: Lightweight, in-memory Server-Sent Events (SSE) broadcaster synchronizing real-time telemetry across all browser clients.

### C. Scientific Validation & Leakage Prevention (`src/`)
- **`test_leakage.py`**: Automated data leakage test validating separation of training, validation, and test splits across temporal and recording-session boundaries.
- **`evaluate.py`**: Detailed classification reports, per-class metrics, confusion matrix rendering, and ROC-AUC curve generation.

### D. Automated 1-Click Launchers
- **`start_all.bat`**: Simultaneously starts the FastAPI backend (port 8000) and Next.js frontend (port 3000) in separate windows.
- **`start_backend.bat`**: Starts only the Uvicorn backend with auto-reload.
- **`start_frontend.bat`**: Starts only the Next.js development server.
- **`push_to_github.bat`**: Automated script to stage, commit, and push all project modules to the collaborative GitHub branch.

---

## 4. Machine Learning & Acoustic Analysis

### Model Architecture (`src/model_v2.py`, `models/best_model_v2.pth`)
- **Base Architecture**: 4-Block Convolutional Neural Network with Squeeze-and-Excitation (SE) Attention blocks.
- **Input Representation**: 128-band Log-Mel Spectrograms extracted from 16 kHz audio (6.0 second window).
- **Target Classes**:
  1. `Roar` (Distress / aggressive calls)
  2. `Rumble` (Low-frequency social communication, 14–24 Hz infra-sound)
  3. `Trumpet` (High-frequency alarm / excitement calls)
  4. `Non-Elephant` (Ambient jungle noise, wind, insects, vehicles)
- **Model Checkpoints**:
  - `models/best_model_v2.pth` (Highest validation accuracy weights)
  - `models/baseline_cnn_v2.pth` (Baseline comparison model)
  - `models/training_curves_v2.png` (Loss & accuracy progression)
  - `models/confusion_matrix.png` (Class-by-class confusion matrix)

---

## 5. GIS, Satellite Map & Organic Polygon Cutouts

The spatial intelligence pipeline features high-precision vectorization and satellite fusion:

| Village | Location | Coordinates | Risk Profile |
| :--- | :--- | :--- | :--- |
| **Sembukarai** | Anaikatti Forest Fringe | `11.1096° N, 76.7621° E` | High elephant corridor crossing |
| **Kandivazhi** | Valley Settlement | `11.1189° N, 76.7715° E` | Agricultural conflict hotspot |
| **Pudur** | Forest Edge | `11.1250° N, 76.7820° E` | Evening transit path |
| **Panapalli** | Water-source Corridor | `11.1340° N, 76.7905° E` | Perennial water body transit |
| **Kodanur** | Foothill Settlement | `11.1420° N, 76.8010° E` | Seasonal migration corridor |

* **Annotation Data**: `data/village_polygon_cutouts/annotations/*.json`
* **Processed Polygons**: `data/village_polygon_cutouts/outputs/`
* **Web GeoJSON Layers**: `web/frontend/data/maconflic-*.geojson`

---

## 6. Backend API Endpoints

The backend is powered by **FastAPI** running on `http://localhost:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and service inventory |
| `GET` | `/docs` | Interactive Swagger UI documentation |
| `POST` | `/api/predict` | Upload audio clip (`.wav`, `.mp3`) → returns classification & Grad-CAM |
| `GET` | `/api/incidents` | Fetch all wildlife sightings with filters (date, severity, type) |
| `POST` | `/api/incidents` | Create a new incident report (GPS, call type, confidence) |
| `GET` | `/api/incidents/heatmap` | GeoJSON FeatureCollection formatted for GIS map layers |
| `GET` | `/api/alerts` | List all active early-warning broadcast zones |
| `POST` | `/api/alerts` | Ranger alert broadcast endpoint |
| `POST` | `/api/alerts/{id}/dismiss` | Deactivate/resolve an active alert zone |
| `GET` | `/api/stats` | Aggregated dashboard statistics (total calls, high risk, active alerts) |
| `GET` | `/api/events/stream` | Server-Sent Events (SSE) live telemetry feed |

---

## 7. Frontend Application Pages

The frontend is a modern **Next.js 16** application with client-side dynamic routing, accessible at `http://localhost:3000`:

### 1. 📊 Dashboard (`/`)
* **Executive KPIs**: Real-time counters for Total Detections, Active Threat Zones, High-Risk Incidents, and Active Sensors.
* **Live Incident Feed**: Chronological table of incoming detections with timestamp, location, call type, confidence, and severity tags.
* **Sensor Telemetry Status**: Health monitoring of deployed microphones and simulated IoT nodes with ping intervals.
* **Quick Navigation & SSE Stream**: Seamless client-side navigation with real-time SSE websocket-style background sync.

### 2. 🗺️ GIS Map (`/map`)
* **Interactive Leaflet Canvas**: Multi-layer map with OpenStreetMap base layer and smooth zoom/pan controls.
* **Incident Clustering & Heatmaps**: Visual density heatmaps highlighting persistent elephant corridors and hotspots.
* **Severity Color-Coding**:
  * 🟢 **Low**: Peaceful sighting, non-threatening vocalization.
  * 🟡 **Medium**: Foraging group near peripheral zone.
  * 🟠 **High**: Elephant herd within 1.0 km buffer of village boundary.
  * 🔴 **Critical**: Imminent crop raid or village crossing within 2.0 km danger radius.
* **Interactive Marker Popups**: Detailed incident cards with timestamp, coordinates, detected call type, and audio confidence.

### 3. 🛰️ Satellite Risk Map (`/map/satellite`)
* **ESRI World Imagery**: High-definition satellite view revealing terrain contours, forest cover, and village borders.
* **Organic Village Polygon Cutouts**: Exact non-convex settlement boundaries for **Sembukarai, Kandivazhi, Pudur, Panapalli, and Kodanur**.
* **Settlement Intelligence Cards**: Population estimates, nearest corridor distances, dynamic risk levels, and recent nearby detections.
* **Dynamic Danger Buffers**: Visualized radius zones centered on active conflict events.

### 4. 🔬 AI Audio Analyzer (`/analyze`)
* **Audio Ingestion**: Drag-and-drop or file upload for `.wav` and `.mp3` acoustic recordings.
* **Multi-Class Prediction**: Real-time inference using the **SE-CNN v2** model classifying calls into *Roar*, *Rumble*, *Trumpet*, or *Non-Elephant*.
* **Confidence Breakdown**: Percentage confidence meters for each class.
* **Grad-CAM Explainability**: Interactive visual heatmap overlaid on the log-mel spectrogram showing exactly which frequency bands (e.g. low-frequency 14–24 Hz rumble vs high-frequency trumpet burst) triggered the AI's classification.

### 5. 📝 Incident Report Portal (`/report`)
Crowdsourcing portal for field rangers, patrol units, and local villagers to log verified sightings:
* **📍 Location Coordinates**:
  * Manual Latitude & Longitude inputs (supports Google Maps coordinates).
  * **"📍 Use Current Location" Button**: One-tap GPS geolocation from mobile/desktop browser.
* **🐘 Incident Details**:
  * **Call Type**: Selectable between *Unknown / Not Heard*, *🦁 Roar (Aggressive)*, *🔊 Rumble (Communication)*, or *🎺 Trumpet (Alarm/Excitement)*.
  * **Severity Level**: Single-click radio selection:
    * 🟢 **Low** — Peaceful sighting.
    * 🟡 **Medium** — Caution needed.
    * 🟠 **High** — Potential conflict.
    * 🔴 **Critical** — Immediate danger.
  * **Description**: Detailed situational notes (e.g. herd size, movement direction, proximity to crops).
* **👤 Reporter Information**: Full Name and Contact Number for verification.
* **Instant System Integration**: Upon clicking **"📤 Submit Incident Report"**, the sighting is committed to the database, evaluated by the Risk & Alert engines, and broadcast live to the GIS and Satellite maps.

### 6. 🔔 Alerts Center (`/alerts`)
* **Active Broadcast Hub**: Real-time registry of all active danger zones across forest fringes.
* **Ranger Broadcast Action**: Direct button for authorized rangers to declare community alerts with designated radius and expiration time.
* **Multi-Channel Dispatch Simulation**: Triggers simulated SMS community warnings and village siren activations.
* **Alert Resolution**: Allows rangers to mark resolved situations and dismiss active zones once elephants return to deep reserve forest.

---

## 8. Directory Structure

```text
ELEPHANT/
├── data/                                 # Spatial datasets, annotations & cutouts
│   ├── village_polygon_cutouts/          # Vector annotations and cutout outputs
│   ├── settlement_outlines/              # Settlement masks and comparison images
│   ├── Seven Villages/                   # High-res village satellite captures
│   └── elephant_observations.geojson     # Verified historical observation records
├── models/                               # Trained neural network weights & figures
│   ├── best_model_v2.pth                 # Production SE-CNN weights
│   ├── baseline_cnn_v2.pth               # Baseline checkpoint
│   ├── confusion_matrix.png              # Multi-class confusion matrix
│   └── training_curves_v2.png            # Loss and accuracy curves
├── reports/                              # Analytical and segmentation reports
├── results/                              # Validation outputs and test metrics
├── scripts/                              # Spatial and pipeline automation scripts
│   ├── auto_generate_organic_polygons.py # Organic polygon geometry generator
│   ├── generate_risk_raster.py           # Spatial risk surface calculator
│   ├── generate_red_corridor.py          # Elephant movement corridor extractor
│   ├── sensor_simulator.py               # Autonomous IoT sensor telemetry daemon
│   └── validate_polygon_cutouts.py       # Geometric consistency validator
├── src/                                  # Core AI/ML pipeline source code
│   ├── model_v2.py                       # SE-CNN PyTorch architecture definition
│   ├── train_v2.py                       # Training loop with early stopping
│   ├── evaluate.py                       # Model evaluation and metrics calculation
│   ├── gradcam.py                        # Grad-CAM heatmap visualization
│   └── test_leakage.py                   # Data leakage validation suite
├── web/
│   ├── backend/                          # FastAPI REST application
│   │   ├── main.py                       # App factory & route handlers
│   │   ├── database.py                   # SQLite schema & query layer
│   │   ├── ml_service.py                 # PyTorch model loading & audio inference
│   │   ├── alert_engine.py               # Automated danger zone calculation
│   │   ├── risk_engine.py                # Multi-criteria risk scoring
│   │   └── event_manager.py              # SSE real-time event publisher
│   └── frontend/                         # Next.js 16 Web Application
│       ├── app/                          # App router pages (map, satellite, analyze...)
│       ├── data/                         # Local GeoJSON spatial assets
│       └── public/                       # Public icons, images, and map overlays
├── start_all.bat                         # 1-Click launcher: Backend + Frontend
├── start_backend.bat                     # 1-Click launcher: FastAPI only
├── start_frontend.bat                    # 1-Click launcher: Next.js only
├── push_to_github.bat                    # 1-Click push script for GitHub collaborator branch
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## 9. How to Run (Quickstart)

### Method 1: 1-Click Launch (Recommended)
Double-click **`start_all.bat`** or run in PowerShell:
```powershell
.\start_all.bat
```
This automatically launches both services in separate terminal windows:
* **Backend**: `http://localhost:8000` (Docs: `http://localhost:8000/docs`)
* **Frontend**: `http://localhost:3000`

---

### Method 2: Manual Terminal Commands

#### 1. Start Backend (Terminal 1)
```powershell
# From the project root folder
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start Frontend (Terminal 2)
```powershell
# In a new terminal window
cd web\frontend
npm run dev
```

#### 3. Start IoT Sensor Simulator (Optional Demo Telemetry)
```powershell
# In a new terminal window
python scripts/sensor_simulator.py
```
*Simulates remote forest microphones transmitting detected elephant calls every 8–15 seconds.*

---

## 10. Analytical Scripts & Pipelines

* **Evaluate Model Performance**:
  ```bash
  python src/evaluate.py
  ```
* **Run Data Leakage Validation**:
  ```bash
  python src/test_leakage.py
  ```
* **Generate Organic Village Cutouts**:
  ```bash
  python scripts/auto_generate_organic_polygons.py
  ```
* **Validate Polygon Geometry**:
  ```bash
  python scripts/validate_polygon_cutouts.py
  ```
* **Generate Movement Risk Raster**:
  ```bash
  python scripts/generate_risk_raster.py
  ```

---

## 11. Git Collaboration & Pushing Changes

The repository is configured with the dedicated feature branch:
* **Branch**: `feature/elephant-system`
* **Remote**: `https://github.com/Kamali-Sree/ELEPHANT.git`

### Automated Push
To push all additions, models, and scripts, run:
```powershell
.\push_to_github.bat
```

### Manual Push Commands
```bash
git checkout -B feature/elephant-system
git add -A
git commit -m "feat: complete elephant conflict monitoring system with models, satellite map, alert/risk engines, and analysis pipeline"
git push -u origin feature/elephant-system
```

---
*MACONFLIC Platform — AI & Spatial Intelligence for Human-Elephant Conflict Mitigation.*

