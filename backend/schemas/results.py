"""Pydantic schemas enforcing the unified JSON response structure for AURA Vision."""

from typing import List, Optional
from pydantic import BaseModel, Field


class InspectionMetadata(BaseModel):
    """Metadata detailing the physical asset and field inspection context."""

    asset_name: str = Field(default="Unnamed Asset", description="Designated name or ID of the infrastructure asset")
    asset_type: str = Field(..., description="Category of asset ('road', 'bridge', 'building', 'concrete')")
    location: str = Field(default="Unspecified Location", description="Geographical or structural location")
    inspector_name: str = Field(default="Field Inspector", description="Name or identifier of the inspecting personnel")
    timestamp: str = Field(..., description="ISO 8601 or formatted inspection timestamp")


class BoundingBox(BaseModel):
    """Coordinates representing a detected damage bounding box [x1, y1, x2, y2]."""

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class DamageItem(BaseModel):
    """Individual damage finding identified by an inspection model."""

    type: str = Field(..., description="Class name of the detected damage or classification")
    confidence: float = Field(..., description="Model detection/classification confidence score (0.0 to 1.0)")
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box coordinates if object detection; null for classification",
    )


class ModelInfo(BaseModel):
    """Metadata regarding the ML model executed for inspection."""

    name: str = Field(..., description="Model checkpoint name")
    task: str = Field(..., description="Inference task type: 'object_detection' or 'image_classification'")


class SeverityResult(BaseModel):
    """Calculated damage severity assessment."""

    level: str = Field(..., description="Severity category: 'Low', 'Medium', 'High', or 'Critical'")
    score: int = Field(..., ge=0, le=100, description="Severity score on a 0-100 scale")


class ConditionResult(BaseModel):
    """Overall structural condition rating derived from severity."""

    score: int = Field(..., ge=0, le=100, description="Overall asset condition score on a 0-100 scale")
    status: str = Field(
        ...,
        description="Condition status label: 'Excellent', 'Good', 'Fair', 'Poor', or 'Critical'",
    )


class RecommendationResult(BaseModel):
    """Actionable domain-specific maintenance guidance."""

    action: str = Field(..., description="Recommended mitigation or inspection action")
    priority: str = Field(..., description="Action priority level: 'Low', 'Medium', 'High', or 'Critical'")


class AnalysisResponse(BaseModel):
    """Unified inspection response strictly adhering to the AURA Vision schema."""

    inspection_metadata: Optional[InspectionMetadata] = Field(
        default=None,
        description="Asset metadata including name, location, inspector, and timestamp",
    )
    asset_type: str = Field(..., description="Analyzed asset type ('road', 'bridge', 'building', 'concrete')")
    model: ModelInfo = Field(..., description="Model information")
    damage: List[DamageItem] = Field(default_factory=list, description="List of detected damage defects")
    severity: SeverityResult = Field(..., description="Computed severity metrics")
    condition: ConditionResult = Field(..., description="Asset condition metrics")
    recommendation: RecommendationResult = Field(..., description="Maintenance recommendation")
    annotated_image: Optional[str] = Field(
        default=None,
        description="Optional Base64-encoded annotated image with overlaid bounding boxes (if applicable)",
    )


# Alias for compatibility with DELTA prompt specification
AnalysisResult = AnalysisResponse


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(default="healthy", description="API operational status")
    version: str = Field(default="1.0.0", description="Application version")
    loaded_models: List[str] = Field(default_factory=list, description="List of active loaded model checkpoints")
