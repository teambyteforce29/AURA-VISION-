"""Specialized inference engine for Concrete & Corrosion Infrastructure Inspection.

Model: AURA_Vision_CONCORNET2023.pt (CONCORNET2023 dataset)
Task: Object Detection
Classes: 0: Corrosion
Output: Defect predictions, bounding boxes, and confidence scores.
"""

from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO

from backend.schemas.results import BoundingBox, DamageItem, ModelInfo


CONCRETE_CLASS_NAMES = {
    0: "Corrosion",
}

MODEL_NAME = "AURA_Vision_CONCORNET2023"
TASK_TYPE = "object_detection"


def run_concornet_inference(
    model: YOLO,
    pil_image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[List[DamageItem], ModelInfo]:
    """Executes corrosion/concrete defect detection on the input image.

    Args:
        model: Preloaded Ultralytics YOLO model instance.
        pil_image: RGB PIL image of concrete structure.
        conf_threshold: Confidence filtering threshold.
        iou_threshold: NMS IoU threshold.

    Returns:
        Tuple[List[DamageItem], ModelInfo]: Detected corrosion damage and model metadata.
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
    names = getattr(result, "names", CONCRETE_CLASS_NAMES)

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()

            raw_label = names.get(cls_id, CONCRETE_CLASS_NAMES.get(cls_id, f"Defect_{cls_id}"))
            class_label = str(raw_label).capitalize()

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
