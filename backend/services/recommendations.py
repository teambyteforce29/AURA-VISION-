"""Actionable Maintenance Recommendations Engine.

Implements domain-specific rules mapping detected damage types and severity
to targeted engineering and maintenance recommendations.
"""

from typing import List
from backend.schemas.results import DamageItem, RecommendationResult, SeverityResult


DOMAIN_RECOMMENDATIONS = {
    # Road damage actions
    "road_crack": "Inspect crack width/depth. Seal minor cracks to prevent water infiltration.",
    "road_pothole": "Clear debris, patch with asphalt, and inspect drainage.",
    "road_surface erosion": "Resurface worn asphalt layer and evaluate sub-base compaction.",
    "road_surface_erosion": "Resurface worn asphalt layer and evaluate sub-base compaction.",

    # Bridge damage actions (DACL10K)
    "bridge_rust": "Remove corrosion, assess structural section loss, apply anti-corrosion coating.",
    "bridge_spalling": "Inspect exposed rebar, patch concrete, evaluate load capacity.",
    "bridge_cavity": "Probe void depth, pressure-grout cavity, and assess structural integrity.",
    "bridge_weathering": "Clean surface, evaluate freeze-thaw degradation, apply water-repellent sealant.",
    "bridge_efflorescence": "Identify moisture source, repair waterproofing membrane, clean mineral deposits.",
    "bridge_crack": "Perform ultrasonic crack depth testing, epoxy-inject active cracks, monitor displacement.",

    # Building damage actions (SDNET2018)
    "building_cracked": "Monitor crack propagation and escalate structural cracking to a certified structural engineer.",
    "building_non_cracked": "No visible structural defects identified. Continue routine periodic observation.",
    "building_non-cracked": "No visible structural defects identified. Continue routine periodic observation.",

    # Concrete / Corrosion
    "concrete_corrosion": "Inspect rebar passivity, remove spalled concrete, treat corroded steel, apply migratory corrosion inhibitors.",
}

# Defect hierarchy for prioritizing multi-defect scenarios
DEFECT_PRIORITY_ORDER = [
    "spalling",
    "cavity",
    "pothole",
    "corrosion",
    "cracked",
    "crack",
    "rust",
    "surface erosion",
    "surface_erosion",
    "weathering",
    "efflorescence",
    "non_cracked",
    "non-cracked",
]


def generate_recommendation(
    asset_type: str,
    damage_items: List[DamageItem],
    severity: SeverityResult,
) -> RecommendationResult:
    """Generates targeted maintenance actions based on detected defects and severity level.

    Args:
        asset_type: Infrastructure asset ('road', 'bridge', 'building', 'concrete').
        damage_items: Detected damages list.
        severity: Computed SeverityResult.

    Returns:
        RecommendationResult: Recommended action text and priority rating.
    """
    clean_asset = asset_type.lower().strip()

    # Case: No defects detected
    if not damage_items:
        return RecommendationResult(
            action="Routine visual inspection passed. Maintain standard maintenance and observation intervals.",
            priority="Low",
        )

    # Determine highest-priority defect present
    defect_names = [item.type.lower().strip() for item in damage_items]
    
    selected_defect = defect_names[0]
    for prio_defect in DEFECT_PRIORITY_ORDER:
        matching = [d for d in defect_names if prio_defect in d]
        if matching:
            selected_defect = matching[0]
            break

    # Look up domain rule
    lookup_key = f"{clean_asset}_{selected_defect}"
    action = DOMAIN_RECOMMENDATIONS.get(lookup_key)

    if not action:
        # Fallback to defect name lookup across generic rules
        for key, text in DOMAIN_RECOMMENDATIONS.items():
            if selected_defect in key:
                action = text
                break

    if not action:
        action = f"Perform localized inspection for {selected_defect} and execute appropriate surface repair."

    # Priority matches calculated severity level (Low, Medium, High, Critical)
    priority = severity.level

    return RecommendationResult(action=action, priority=priority)
