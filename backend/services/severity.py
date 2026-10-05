"""Severity Calculation Engine.

Disambiguates model detection confidence from physical structural defect severity.
Computes a normalized severity score (0-100) and categorizes it into:
Low, Medium, High, or Critical.
"""

from typing import List, Tuple
from backend.schemas.results import DamageItem, SeverityResult


# Relative hazard weights for various defect types
DEFECT_WEIGHTS = {
    # Bridge defects
    "spalling": 1.5,       # Structural concrete loss, exposed rebar risk
    "cavity": 1.4,         # Internal void or substantial pocket
    "crack": 1.3,          # Fracture / stress propagation
    "rust": 1.1,           # Corrosion of steel / rebar section loss
    "weathering": 0.8,     # Surface environmental wear
    "efflorescence": 0.6,  # Salt deposits indicating moisture intrusion
    
    # Road defects
    "pothole": 1.5,        # Immediate vehicle hazard and sub-base failure
    "surface erosion": 0.9,# Pavement wear and raveling
    
    # Concrete / General
    "corrosion": 1.3,      # Active corrosion of structural element
    
    # Building classification
    "cracked": 1.2,
    "non_cracked": 0.0,
    "non-cracked": 0.0,
}


def calculate_severity(
    damage_items: List[DamageItem],
    image_width: int,
    image_height: int,
    asset_type: str,
    is_classification: bool = False,
) -> SeverityResult:
    """Calculates the physical defect severity score (0-100) and categorized level.

    Formulation considerations:
    1. Defect Type Weight (hazard severity of the specific failure mode)
    2. Bounding Box Coverage Ratio (percentage of structural surface damaged)
    3. Defect Count (density and dispersion of defects)
    4. Model Confidence (confidence scaling to avoid over-penalizing uncertain detections)

    Args:
        damage_items: Detections or classification results.
        image_width: Width of the input image in pixels.
        image_height: Height of the input image in pixels.
        asset_type: Asset category ('road', 'bridge', 'building', 'concrete').
        is_classification: True if the model output is single-label classification.

    Returns:
        SeverityResult: Object containing categorized level and integer score (0-100).
    """
    if not damage_items:
        return SeverityResult(level="Low", score=0)

    # Classification handling (e.g. Building Crack: cracked vs non_cracked)
    if is_classification:
        top_item = damage_items[0]
        label = top_item.type.lower().replace("-", "_")
        conf = float(top_item.confidence)

        if "non" in label:
            # Clean structure
            score = max(0, min(15, int(round((1.0 - conf) * 15))))
            level = "Low"
            return SeverityResult(level=level, score=score)
        else:
            # Cracked wall/surface - severity scales with detection certainty
            # Base severity for visible crack is 55, scaling up to 85 with confidence
            score = int(round(55.0 + (conf * 30.0)))
            score = max(30, min(95, score))
            level = categorize_score(score)
            return SeverityResult(level=level, score=score)

    # Object Detection severity calculation (Road, Bridge, Concrete)
    total_image_area = max(1.0, float(image_width * image_height))
    weighted_damage_area = 0.0
    defect_count_score = 0.0
    max_individual_weight = 0.0

    for item in damage_items:
        clean_name = item.type.lower().strip()
        weight = DEFECT_WEIGHTS.get(clean_name, 1.0)
        max_individual_weight = max(max_individual_weight, weight)
        conf = max(0.2, min(1.0, float(item.confidence)))

        if item.bounding_box:
            bbox = item.bounding_box
            w = max(0.0, bbox.x2 - bbox.x1)
            h = max(0.0, bbox.y2 - bbox.y1)
            box_area = w * h
            coverage_fraction = min(1.0, box_area / total_image_area)
            # Area component weighted by defect hazard and confidence
            weighted_damage_area += coverage_fraction * weight * conf
        else:
            # Fallback if no bbox provided for detection
            weighted_damage_area += 0.05 * weight * conf

        # Count component (diminishing return per additional defect)
        defect_count_score += 8.0 * weight * conf

    # Convert area coverage to a 0-50 component
    # e.g., 15% image coverage of high-weight defect yields ~40 points
    area_component = min(50.0, weighted_damage_area * 250.0)

    # Convert defect count to a 0-35 component
    count_component = min(35.0, defect_count_score)

    # Base hazard factor from worst defect detected (0-15 points)
    base_hazard_component = min(15.0, (max_individual_weight - 0.5) * 15.0)

    raw_score = area_component + count_component + base_hazard_component
    final_score = int(round(max(5, min(100, raw_score))))

    level = categorize_score(final_score)
    return SeverityResult(level=level, score=final_score)


def categorize_score(score: int) -> str:
    """Categorizes a 0-100 severity score into standard inspection levels."""
    if score >= 75:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 25:
        return "Medium"
    else:
        return "Low"
