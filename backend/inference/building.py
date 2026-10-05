"""Specialized inference engine for Building Wall Crack Classification.

Model: AURA_Vision_Building_Crack_YOLO11s.pt (SDNET2018 dataset)
Task: Image Classification
Classes: Cracked, Non-cracked
Output: Class name and top-1 confidence score.
Constraint: DO NOT compute or render bounding boxes.
"""

from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO

from backend.schemas.results import DamageItem, ModelInfo


MODEL_NAME = "AURA_Vision_Building_Crack_YOLO11s"
TASK_TYPE = "image_classification"

DISPLAY_NAMES = {
    0: "Cracked",
    1: "Non-cracked",
    "cracked": "Cracked",
    "non_cracked": "Non-cracked",
    "non-cracked": "Non-cracked",
}


def run_building_inference(
    model: YOLO,
    pil_image: Image.Image,
) -> Tuple[List[DamageItem], ModelInfo]:
    """Executes whole-image crack classification on building wall surfaces.

    Strictly produces classification results without bounding boxes.

    Args:
        model: Preloaded Ultralytics YOLO classification model.
        pil_image: RGB PIL image of building wall.

    Returns:
        Tuple[List[DamageItem], ModelInfo]: Classification damage item and model info.
    """
    model_info = ModelInfo(name=MODEL_NAME, task=TASK_TYPE)
    damage_items: List[DamageItem] = []

    # Execute classification inference (image size 224 as per model training)
    results = model.predict(
        source=pil_image,
        imgsz=224,
        verbose=False,
    )

    if not results or len(results) == 0:
        return damage_items, model_info

    result = results[0]

    if result.probs is not None:
        top1_idx = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf.item())

        raw_name = result.names.get(top1_idx, DISPLAY_NAMES.get(top1_idx, f"Class_{top1_idx}"))
        formatted_name = DISPLAY_NAMES.get(str(raw_name).lower(), str(raw_name).capitalize())

        # DO NOT compute or render bounding boxes for classification
        damage_items.append(
            DamageItem(
                type=formatted_name,
                confidence=round(top1_conf, 4),
                bounding_box=None,
            )
        )

    return damage_items, model_info
