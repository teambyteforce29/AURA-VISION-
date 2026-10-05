"""Specialized inference engine for Bridge Infrastructure Damage Inspection.

Model: AURA_Vision_Bridge_Damage_YOLO11s.pt (DACL10K dataset)
Task: Object Detection
Classes: 0: Rust, 1: Spalling, 2: Cavity, 3: Weathering, 4: Efflorescence, 5: Crack
Output: Bounding boxes [x1, y1, x2, y2], class names, confidence scores.
Note: Positioned as an inspection-support / screening tool.
"""

from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO

from backend.schemas.results import BoundingBox, DamageItem, ModelInfo


BRIDGE_CLASS_NAMES = {
    0: "Rust",
    1: "Spalling",
    2: "Cavity",
    3: "Weathering",
    4: "Efflorescence",
    5: "Crack",
}

MODEL_NAME = "AURA_Vision_Bridge_Damage_YOLO11s"
TASK_TYPE = "object_detection"


def run_bridge_inference(
    model: YOLO,
    pil_image: Image.Image,
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.45,
) -> Tuple[List[DamageItem], ModelInfo]:
    """Executes bridge damage object detection screening on the input image.

    Args:
        model: Preloaded Ultralytics YOLO model instance.
        pil_image: RGB PIL image of bridge element/structure.
        conf_threshold: Confidence filtering threshold.
        iou_threshold: NMS IoU threshold.

    Returns:
        Tuple[List[DamageItem], ModelInfo]: Detected bridge defects and model metadata.
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
    names = getattr(result, "names", BRIDGE_CLASS_NAMES)

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()

            class_label = names.get(cls_id, BRIDGE_CLASS_NAMES.get(cls_id, f"BridgeDefect_{cls_id}"))

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
