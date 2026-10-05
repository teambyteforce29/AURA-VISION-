"""FastAPI application entrypoint for AURA Vision.

Provides endpoints:
- GET  /api/health   : Health status & list of preloaded models
- POST /api/analyze  : AI damage analysis across road, bridge, building, concrete
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from backend.schemas.results import AnalysisResponse, HealthResponse
from backend.utils.image import validate_image_bytes
from backend.inference.router import process_inspection, VALID_ASSET_TYPES


# Model checkpoints mapping to their respective filenames in the models/ directory
MODEL_FILENAMES = {
    "road": "AURA_Vision_Road_Damage_YOLO11s.pt",
    "bridge": "AURA_Vision_Bridge_Damage_YOLO11s.pt",
    "building": "AURA_Vision_Building_Crack_YOLO11s.pt",
    "concrete": "AURA_Vision_CONCORNET2023.pt",
}

# In-memory singleton registry for preloaded YOLO models
loaded_models: Dict[str, YOLO] = {}


def resolve_models_dir() -> Path:
    """Finds the models directory across development environments."""
    # First priority: relative to backend/ (AURA_Vision/models)
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    if models_dir.exists():
        return models_dir

    # Second priority: current working directory
    cwd_models = Path.cwd() / "models"
    if cwd_models.exists():
        return cwd_models

    return models_dir


def load_all_models() -> Dict[str, YOLO]:
    """Loads all 4 PyTorch/Ultralytics YOLO .pt model files once at startup."""
    models_dir = resolve_models_dir()
    print(f"[AURA Vision] Loading models from: {models_dir}")

    registry = {}
    for asset_type, filename in MODEL_FILENAMES.items():
        model_path = models_dir / filename
        if not model_path.exists():
            print(f"[WARNING] Model checkpoint not found at: {model_path}")
            continue

        try:
            print(f"[AURA Vision] Initializing {asset_type.upper()} model ({filename})...")
            # Load PyTorch YOLO weights into memory once
            model = YOLO(str(model_path))
            registry[asset_type] = model
            print(f"[AURA Vision] Successfully loaded {asset_type.upper()} model.")
        except Exception as e:
            print(f"[ERROR] Failed to load model for {asset_type} ({filename}): {e}")

    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager ensuring singleton model loading ONCE at startup."""
    global loaded_models
    loaded_models = load_all_models()
    app.state.models = loaded_models
    print(f"[AURA Vision] System startup complete. Active models: {list(loaded_models.keys())}")
    yield
    print("[AURA Vision] Shutting down application...")
    loaded_models.clear()


# Initialize FastAPI with metadata and lifespan handler
app = FastAPI(
    title="AURA Vision API",
    description="AI-powered infrastructure damage inspection and maintenance recommendations API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS middleware for all origins to facilitate frontend and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns the operational status and list of preloaded model checkpoints."""
    active_keys = list(loaded_models.keys())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        loaded_models=active_keys,
    )


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["Inference"])
async def analyze_damage(
    asset_type: str = Form(..., description="Asset type: 'road', 'bridge', 'building', or 'concrete'"),
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    asset_name: Optional[str] = Form(default="Unnamed Asset", description="Infrastructure asset designation or ID"),
    location: Optional[str] = Form(default="Unspecified Location", description="Geographical or structural location"),
    inspector_name: Optional[str] = Form(default="Field Inspector", description="Name of inspecting personnel"),
    timestamp: Optional[str] = Form(default=None, description="Optional custom timestamp (ISO 8601 or string)"),
):
    """Accepts an infrastructure image, asset type, and metadata, executing specialized AI damage detection.

    Returns the unified JSON analysis response including inspection metadata, detected defects,
    severity calculations, structural condition scoring, and maintenance actions.
    """
    clean_asset = asset_type.lower().strip()
    if clean_asset not in VALID_ASSET_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid asset_type '{asset_type}'. Supported types: {sorted(list(VALID_ASSET_TYPES))}",
        )

    # Ensure model is ready
    if clean_asset not in loaded_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model for asset '{clean_asset}' is not loaded or available on server.",
        )

    # Read and validate image file
    try:
        image_bytes = await file.read()
        pil_image = validate_image_bytes(image_bytes)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid image file: {str(val_err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded file: {str(exc)}",
        )

    # Execute pipeline through the unified router
    try:
        response = process_inspection(
            asset_type=clean_asset,
            pil_image=pil_image,
            model_registry=loaded_models,
            asset_name=asset_name,
            location=location,
            inspector_name=inspector_name,
            timestamp=timestamp,
        )
        return response
    except Exception as e:
        print(f"[ERROR] Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing inspection pipeline: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
