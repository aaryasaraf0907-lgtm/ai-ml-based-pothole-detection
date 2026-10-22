# AI/ML-Based Pothole Detection

An AI-powered road monitoring system that uses computer vision to detect potholes, classify their severity, record observations with location metadata, consolidate evidence across buses, and support road maintenance and post-repair verification.

## Features

- **Real-Time Detection:** Detect potholes using a trained YOLO-based computer vision model.
- **Severity Classification:** Classify detected potholes as minor, moderate, or severe.
- **GPS-Tagged Reporting:** Record detection observations with location and bus metadata.
- **Fleet Consensus:** Consolidate observations from multiple buses and routes into persistent road-defect records.
- **Maintenance Prioritization:** Analyze road defects and generate maintenance priorities and recommended actions.
- **Post-Repair Verification:** Track clear verification passes from multiple buses before resolving a repaired defect.
- **Interactive Dashboard:** Review detections, road defects, maintenance status, and reports.

## Technology Stack

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Firebase Admin SDK
- Cloud Firestore
- HTML, CSS, JavaScript, and Leaflet

## Repository Structure

```text
ai-ml-based-pothole-detection/
├── detect_live.py
├── fleet_demo.py
├── maintenance_engine.py
├── mark_repair_completed.py
├── post_repair_verify.py
├── end.html
├── requirements.txt
└── runs/
    └── pothole_severity_v2/
        └── weights/
            └── best.pt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aaryasaraf0907-lgtm/ai-ml-based-pothole-detection.git
cd ai-ml-based-pothole-detection
```

### 2. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Firebase

Configure a Firebase project with Cloud Firestore access. Place your service-account JSON file in the project root as `serviceAccountKey.json`.

**Never commit or publish this credential.**

### 4. Configure the dashboard

Open `end.html` and configure your Firebase web settings if required. Replace `YOUR_CARTO_BASEMAPS_API_KEY` in the basemap URL with your own authorized CARTO key.

## Run Live Pothole Detection

```bash
python detect_live.py --gps mock --camera 0
```

The detector processes webcam frames, displays detections and severity information, and uploads accepted observations to Firestore. Press `Q` in the detector window to exit.

Mock GPS is intended for controlled testing; it does not represent the actual location of the camera.

## Fleet and Maintenance Workflow

Run the fleet-consensus scenario:

```bash
python fleet_demo.py
```

Analyze maintenance requirements:

```bash
python maintenance_engine.py
```

Mark an existing road defect as repaired:

```bash
python mark_repair_completed.py RD-002
```

Replace `RD-002` with an ID that exists in your Firestore database.

Run post-repair verification:

```bash
python post_repair_verify.py --road-defect-id RD-002 --gps defect --bus-id BUS-023
```

Repeat verification with distinct bus IDs. The workflow requires successful clear verification passes from three different buses before a defect can be resolved.

These scripts operate on Firestore data. Review the target records before running maintenance or repair commands.

## Model

The trained inference checkpoint is located at:

`runs/pothole_severity_v2/weights/best.pt`

Ensure the model file is present before running detection.

## Security

- Never commit Firebase service-account credentials, private keys, or secret tokens.
- Review staged files before every Git commit.
- Apply appropriate Firestore security rules and least-privilege permissions.
- Restrict map-provider keys where supported.
- Do not publish sensitive user data or private detection records.

## Project Scope

This repository contains the pothole detection, fleet road-defect intelligence, maintenance, and post-repair verification components of TransitEye. Separate traffic-density and ANPR modules are not included.
