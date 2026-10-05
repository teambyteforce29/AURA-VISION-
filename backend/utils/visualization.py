"""Visualization utilities for rendering bounding boxes and damage tags on infrastructure images."""

from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

from backend.schemas.results import DamageItem
from backend.utils.image import pil_to_cv2, cv2_to_pil


# Color palette for defect classes (BGR format for OpenCV)
CLASS_COLORS = {
    # Road defect classes
    "crack": (40, 120, 240),          # Bright Orange / Blue in BGR
    "pothole": (30, 30, 220),         # Vibrant Crimson Red
    "surface erosion": (0, 215, 255), # Amber / Gold
    
    # Bridge defect classes (DACL10K)
    "rust": (20, 90, 200),            # Rust Orange / Brown
    "spalling": (180, 50, 160),       # Magenta / Purple
    "cavity": (0, 0, 180),            # Deep Red
    "weathering": (200, 160, 40),     # Slate Blue
    "efflorescence": (180, 200, 50),  # Turquoise / Lime
    
    # Concrete / Corrosion
    "corrosion": (0, 69, 255),        # Orange Red
    
    # Fallback
    "default": (0, 165, 255),
}


def get_color_for_class(class_name: str) -> tuple:
    """Returns the BGR color tuple for a given class name."""
    clean_name = class_name.lower().strip()
    return CLASS_COLORS.get(clean_name, CLASS_COLORS["default"])


def render_detections(
    pil_image: Image.Image,
    damage_items: List[DamageItem],
    is_classification: bool = False,
) -> Image.Image:
    """Renders bounding boxes and label tags onto the image for object detection models.

    For classification tasks (e.g. Building Crack), returns the original image untouched.

    Args:
        pil_image: Original RGB PIL image.
        damage_items: List of detected damage items with bounding boxes.
        is_classification: Boolean flag indicating if model was classification.

    Returns:
        PIL.Image.Image: The annotated image (or original untouched image if classification).
    """
    # Specification rule: For Classification (Building), return the original image untouched.
    if is_classification or not damage_items:
        return pil_image

    img_bgr = pil_to_cv2(pil_image)
    img_h, img_w = img_bgr.shape[:2]

    # Compute adaptive scale based on image dimensions
    scale = max(0.4, min(img_w, img_h) / 800.0)
    thickness = max(2, int(round(scale * 2.5)))
    font_scale = max(0.45, scale * 0.55)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for item in damage_items:
        if not item.bounding_box:
            continue

        bbox = item.bounding_box
        x1 = int(round(bbox.x1))
        y1 = int(round(bbox.y1))
        x2 = int(round(bbox.x2))
        y2 = int(round(bbox.y2))

        # Clamp coordinates to image boundaries
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        color = get_color_for_class(item.type)

        # Draw main bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Subtle semi-transparent interior overlay for enhanced visual depth
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, img_bgr, 0.88, 0, img_bgr)

        # Prepare label pill: "Class XX%"
        conf_pct = int(round(item.confidence * 100))
        label_text = f"{item.type.upper()} {conf_pct}%"

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, max(1, int(thickness // 2)))
        pad_x, pad_y = int(6 * scale), int(4 * scale)

        # Calculate label tag position (above box, or inside if near top edge)
        if y1 - th - (pad_y * 2) > 0:
            tag_y1 = y1 - th - (pad_y * 2)
            tag_y2 = y1
        else:
            tag_y1 = y1
            tag_y2 = y1 + th + (pad_y * 2)

        tag_x1 = x1
        tag_x2 = min(img_w - 1, x1 + tw + (pad_x * 2))

        # Draw solid tag background pill
        cv2.rectangle(img_bgr, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
        # Draw tag border
        cv2.rectangle(img_bgr, (tag_x1, tag_y1), (tag_x2, tag_y2), (255, 255, 255), 1, cv2.LINE_AA)

        # Draw white crisp text
        text_pos = (tag_x1 + pad_x, tag_y2 - pad_y - baseline // 2)
        cv2.putText(
            img_bgr,
            label_text,
            text_pos,
            font,
            font_scale,
            (255, 255, 255),
            max(1, int(thickness // 2)),
            cv2.LINE_AA,
        )

    return cv2_to_pil(img_bgr)
