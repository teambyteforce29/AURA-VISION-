"""Inference Router dispatches requests to the appropriate specialized engine

Coordinates ML prediction, severity calculation, condition scoring,
action recommendation, and visualization rendering.
"""

from typing import Dict, Optional
from PIL import Image
from ultralytics import YOLO

from backend.schemas.results import AnalysisResponse
from backend.services.severity import calculate_severity
from backend.services.condition import calculate_condition
from backend.services.recommendations import generate_recommendation
from backend.utils.visualization import render_detections
from backend.utils.image import image_to_base64
from backend.inference.road import run_road_inference
from backend.inference.bridge import run_bridge_inference
from backend.inference.building import run_building_inference
from backend.inference.concornet import run_concornet_inference


from datetime import datetime
from backend.schemas.results import AnalysisResponse, InspectionMetadata


VALID_ASSET_TYPES = {"road", "bridge", "building", "concrete"}


def process_inspection(
    asset_type: str,
    pil_image: Image.Image,
    model_registry: Dict[str, YOLO],
    conf_threshold: Optional[float] = None,
    asset_name: Optional[str] = None,
    location: Optional[str] = None,
    inspector_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    inspection_metadata: Optional[InspectionMetadata] = None,
) -> AnalysisResponse:
    """Dispatches the image to the corresponding specialized model and executes the full assessment pipeline.

    Args:
        asset_type: 'road', 'bridge', 'building', or 'concrete'.
        pil_image: Validated RGB PIL Image.
        model_registry: Mapping of asset type to loaded YOLO model instance.
        conf_threshold: Optional override for confidence threshold.
        asset_name: Optional infrastructure asset designation.
        location: Optional geographical or structural location.
        inspector_name: Optional inspecting engineer/inspector name.
        timestamp: Optional inspection timestamp.
        inspection_metadata: Optional pre-assembled InspectionMetadata instance.

    Returns:
        AnalysisResponse: Unified inspection response conforming strictly to schema.
    """
    clean_asset = asset_type.lower().strip()
    if clean_asset not in VALID_ASSET_TYPES:
        raise ValueError(
            f"Invalid asset_type '{asset_type}'. Must be one of: {', '.join(sorted(VALID_ASSET_TYPES))}"
        )

    model = model_registry.get(clean_asset)
    if model is None:
        raise RuntimeError(
            f"Model for asset '{clean_asset}' is not initialized or loaded in the model registry."
        )

    img_w, img_h = pil_image.size
    is_classification = (clean_asset == "building")

    # 1. Specialized Model Inference
    if clean_asset == "road":
        thresh = conf_threshold if conf_threshold is not None else 0.25
        damage_items, model_info = run_road_inference(model, pil_image, conf_threshold=thresh)
    elif clean_asset == "bridge":
        thresh = conf_threshold if conf_threshold is not None else 0.20
        damage_items, model_info = run_bridge_inference(model, pil_image, conf_threshold=thresh)
    elif clean_asset == "building":
        damage_items, model_info = run_building_inference(model, pil_image)
    elif clean_asset == "concrete":
        thresh = conf_threshold if conf_threshold is not None else 0.25
        damage_items, model_info = run_concornet_inference(model, pil_image, conf_threshold=thresh)
    else:
        raise ValueError(f"Unhandled asset type: {clean_asset}")

    # 2. Severity Calculation Engine
    severity = calculate_severity(
        damage_items=damage_items,
        image_width=img_w,
        image_height=img_h,
        asset_type=clean_asset,
        is_classification=is_classification,
    )

    # 3. Condition Scoring Engine
    condition = calculate_condition(
        severity=severity,
        defect_count=len(damage_items),
    )

    # 4. Actionable Maintenance Recommendation Engine
    recommendation = generate_recommendation(
        asset_type=clean_asset,
        damage_items=damage_items,
        severity=severity,
    )

    # 5. Visualization Utility
    annotated_pil = render_detections(
        pil_image=pil_image,
        damage_items=damage_items,
        is_classification=is_classification,
    )
    annotated_b64 = image_to_base64(annotated_pil)

    # 6. Assemble Inspection Metadata
    if inspection_metadata is not None:
        final_meta = inspection_metadata
    else:
        now_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_meta = InspectionMetadata(
            asset_name=asset_name or f"{clean_asset.capitalize()} Structure",
            asset_type=clean_asset,
            location=location or "Unspecified Location",
            inspector_name=inspector_name or "Field Inspector",
            timestamp=now_str,
        )

    # 7. Assemble Unified Response
    return AnalysisResponse(
        inspection_metadata=final_meta,
        asset_type=clean_asset,
        model=model_info,
        damage=damage_items,
        severity=severity,
        condition=condition,
        recommendation=recommendation,
        annotated_image=annotated_b64,
    )
