# AURA Vision — AI Infrastructure Damage Inspection & Maintenance Recommendations

AURA Vision is an end-to-end, production-ready computer vision application designed to inspect infrastructure assets (roads, bridges, buildings, concrete structures), estimate structural defect severity, score overall asset condition, and generate actionable engineering maintenance recommendations.

---

## 1. System Architecture

```text
                                USER
                                 │
                                 ▼
                     Select Asset Category
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
            ROAD               BRIDGE            BUILDING
              │                  │                  │
              ▼                  ▼                  ▼
           YOLO11s            YOLO11s            YOLO11s
       Object Detection   Object Detection    Classification
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                        CONCRETE / CORROSION
                             (YOLO11s)
                                 │
                                 ▼
                      Unified Damage Analysis
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      Defect Detection       Severity           Condition
      & Bounding Boxes       Analysis             Score
                                 │
                                 ▼
                      Actionable Recommendation
                                 │
                                 ▼
                     INSPECTION REPORT & DASHBOARD
```

---

## 2. Directory Layout

```text
AURA_Vision/
├── backend/
│   ├── main.py                  # FastAPI app with CORS and startup model preloading
│   ├── inference/
│   │   ├── __init__.py          # Exports inference engines
│   │   ├── router.py            # Central inference dispatcher & coordinator
│   │   ├── road.py              # Road damage detection (Crack, Pothole, Surface Erosion)
│   │   ├── bridge.py            # Bridge damage screening (Rust, Spalling, Cavity, etc.)
│   │   ├── building.py          # Building crack classification (Cracked vs Non-cracked)
│   │   └── concornet.py         # Concrete corrosion detection (CONCORNET2023)
│   ├── services/
│   │   ├── severity.py          # Disambiguates confidence from defect severity (0-100)
│   │   ├── condition.py         # Calibrates overall asset condition score (0-100)
│   │   ├── recommendations.py   # Domain-specific actionable maintenance rules
│   │   └── reports.py           # Multi-format report export (PDF, TXT, JSON)
│   ├── schemas/
│   │   └── results.py           # Unified Pydantic output response schemas (with InspectionMetadata)
│   └── utils/
│       ├── image.py             # Image validation, decoding, Base64 conversion
│       └── visualization.py     # Clean bounding box & pill label rendering
├── models/                        # Four required model checkpoints, stored with Git LFS
├── frontend/
│   └── app.py                   # Streamlit dashboard with metadata inputs & multi-format exports
├── requirements.txt             # Python dependencies (including fpdf2)
└── README.md                    # System documentation
```

---

## 3. The 4 Specialized ML Models

| Asset Type | Model File | Task | Classes | Key Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Road** | `AURA_Vision_Road_Damage_YOLO11s.pt` | Object Detection | `Crack`, `Pothole`, `Surface Erosion` | Road Surface Dataset |
| **Bridge** | `AURA_Vision_Bridge_Damage_YOLO11s.pt` | Object Detection | `Rust`, `Spalling`, `Cavity`, `Weathering`, `Efflorescence`, `Crack` | DACL10K |
| **Building** | `AURA_Vision_Building_Crack_YOLO11s.pt` | Image Classification | `Cracked`, `Non-cracked` | SDNET2018 (Walls) |
| **Concrete** | `AURA_Vision_CONCORNET2023.pt` | Object Detection | `Corrosion` | CONCORNET2023 |

The four model checkpoints are stored with Git LFS. Install and enable Git LFS before cloning so the model files are downloaded and available in `models/` for inference.

> **Important Visualization Note:** For Road, Bridge, and Concrete, bounding boxes and confidence pills are rendered directly on the image. For Building Crack classification, the original image is returned untouched without artificial bounding boxes.

---

## 4. Unified JSON Output Schema

All models return a standardized response format:

```json
{
  "inspection_metadata": {
    "asset_name": "Overpass Viaduct #21",
    "asset_type": "bridge",
    "location": "Northbound Mile 48.2",
    "inspector_name": "Marcus Vance, PE",
    "timestamp": "2026-10-05 14:30:00"
  },
  "asset_type": "bridge",
  "model": {
    "name": "AURA_Vision_Bridge_Damage_YOLO11s",
    "task": "object_detection"
  },
  "damage": [
    {
      "type": "Rust",
      "confidence": 0.87,
      "bounding_box": {
        "x1": 120.0,
        "y1": 85.0,
        "x2": 420.0,
        "y2": 310.0
      }
    }
  ],
  "severity": {
    "level": "High",
    "score": 78
  },
  "condition": {
    "score": 58,
    "status": "Needs Maintenance"
  },
  "recommendation": {
    "action": "Remove corrosion, assess structural section loss, apply anti-corrosion coating.",
    "priority": "High"
  },
  "annotated_image": "data:image/jpeg;base64,..."
}
```

---

## 5. Severity & Condition Scoring

- **Confidence vs. Severity**: Detection confidence represents the model's certainty that a visual pattern matches a class. Defect severity represents the physical risk to structural integrity.
- **Severity Score (0–100)**: Evaluates bounding box coverage area, defect count, defect hazard weights (e.g. spalling/cavity > weathering), and confidence. Categorized as **Low**, **Medium**, **High**, or **Critical**.
- **Condition Score (0–100)**:
  - `80–100`: **Excellent**
  - `60–79`: **Good**
  - `40–59`: **Fair** (or **Needs Maintenance**)
  - `20–39`: **Poor**
  - `0–19`: **Critical**

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10 or 3.11+
- CUDA-compatible GPU (optional, CPU inference supported out of the box)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: `http://localhost:8000/docs`

### Step 3: Run the Streamlit Frontend
In a separate terminal:
```bash
streamlit run frontend/app.py
```
Open your browser at: `http://localhost:8501`

---

## 7. API Endpoints

### Health Check
```bash
curl -X GET http://localhost:8000/api/health
```

### Analyze Damage
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "asset_type=bridge" \
  -F "file=@sample_bridge.jpg"
```

---

## 8. Disclaimer

**DISCLAIMER**: AURA Vision is an AI-assisted screening and decision-support tool. It is not an automated certification system. All flagged structural anomalies should be validated by certified civil or structural engineers before making critical life-safety or load-bearing operational decisions.
