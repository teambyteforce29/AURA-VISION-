"""Specialized inference engine for Road Infrastructure Damage Inspection.

Model: AURA_Vision_Road_Damage_YOLO11s.pt
Task: Object Detection
Classes: 0: Crack, 1: Pothole, 2: Surface Erosion
Output: Bounding boxes [x1, y1, x2, y2], class names, confidence scores.
"""

from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO

from backend.schemas.results import BoundingBox, DamageItem, ModelInfo


# Default class mapping as verified from checkpoint metadata
ROAD_CLASS_NAMES = {
    0: "Crack",
    1: "Pothole",
    2: "Surface Erosion",
}

MODEL_NAME = "AURA_Vision_Road_Damage_YOLO11s"
TASK_TYPE = "object_detection"


def run_road_inference(
    model: YOLO,
    pil_image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[List[DamageItem], ModelInfo]:
    """Executes road damage object detection on the input image.

    Args:
        model: Preloaded Ultralytics YOLO model instance.
        pil_image: RGB PIL image of road surface.
        conf_threshold: Confidence filtering threshold.
        iou_threshold: NMS IoU threshold.

    Returns:
        Tuple[List[DamageItem], ModelInfo]: Detected damage items and model metadata.
    """
    model_info = ModelInfo(name=MODEL_NAME, task=TASK_TYPE)
    damage_items: List[DamageItem] = []

    # Execute inference
    results = model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    if not results or len(results) == 0:
        return damage_items, model_info

    result = results[0]
    names = getattr(result, "names", ROAD_CLASS_NAMES)

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            class_label = names.get(cls_id, ROAD_CLASS_NAMES.get(cls_id, f"Defect_{cls_id}"))

            damage_items.append(
                DamageItem(
                    type=class_label,
                    confidence=round(conf, 4),
                    bounding_box=BoundingBox(
                        x1=round(coords[0], 2),
                        y1=round(coords[1], 2),
                        x2=round(coords[2], 2),
                        y2=round(coords[3], 2),
                    ),
                )
            )

    return damage_items, model_info
