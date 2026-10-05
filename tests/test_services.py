"""Unit tests for AURA Vision schemas and services."""

import unittest
from backend.schemas.results import (
    BoundingBox,
    DamageItem,
    ModelInfo,
    SeverityResult,
    ConditionResult,
    RecommendationResult,
    AnalysisResponse,
)
from backend.services.severity import calculate_severity, categorize_score
from backend.services.condition import calculate_condition
from backend.services.recommendations import generate_recommendation


class TestSchemas(unittest.TestCase):
    def test_bounding_box(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0)
        self.assertEqual(bbox.x1, 10.0)
        self.assertEqual(bbox.y2, 150.0)

    def test_analysis_response_schema(self):
        resp = AnalysisResponse(
            asset_type="bridge",
            model=ModelInfo(name="AURA_Vision_Bridge_Damage_YOLO11s", task="object_detection"),
            damage=[
                DamageItem(
                    type="Rust",
                    confidence=0.87,
                    bounding_box=BoundingBox(x1=120, y1=85, x2=420, y2=310),
                )
            ],
            severity=SeverityResult(level="High", score=78),
            condition=ConditionResult(score=58, status="Needs Maintenance"),
            recommendation=RecommendationResult(
                action="Inspect corrosion depth and apply protective coating",
                priority="High",
            ),
        )
        self.assertEqual(resp.asset_type, "bridge")
        self.assertEqual(len(resp.damage), 1)
        self.assertEqual(resp.damage[0].type, "Rust")
        self.assertEqual(resp.severity.level, "High")
        self.assertEqual(resp.condition.status, "Needs Maintenance")


class TestSeverityAndCondition(unittest.TestCase):
    def test_clean_asset(self):
        sev = calculate_severity([], 800, 600, "road")
        self.assertEqual(sev.level, "Low")
        self.assertEqual(sev.score, 0)

        cond = calculate_condition(sev, defect_count=0)
        self.assertEqual(cond.score, 100)
        self.assertEqual(cond.status, "Excellent")

    def test_road_pothole_severity(self):
        items = [
            DamageItem(
                type="Pothole",
                confidence=0.92,
                bounding_box=BoundingBox(x1=100, y1=200, x2=350, y2=450),
            )
        ]
        sev = calculate_severity(items, 800, 600, "road")
        self.assertIn(sev.level, ["Medium", "High", "Critical"])
        self.assertGreater(sev.score, 20)

        cond = calculate_condition(sev, defect_count=len(items))
        self.assertLess(cond.score, 85)

    def test_building_classification_cracked(self):
        items = [
            DamageItem(
                type="Cracked",
                confidence=0.95,
                bounding_box=None,
            )
        ]
        sev = calculate_severity(items, 800, 600, "building", is_classification=True)
        self.assertIn(sev.level, ["High", "Critical"])
        self.assertGreaterEqual(sev.score, 70)

        rec = generate_recommendation("building", items, sev)
        self.assertIn("structural engineer", rec.action.lower())

    def test_building_classification_non_cracked(self):
        items = [
            DamageItem(
                type="Non-cracked",
                confidence=0.98,
                bounding_box=None,
            )
        ]
        sev = calculate_severity(items, 800, 600, "building", is_classification=True)
        self.assertEqual(sev.level, "Low")
        self.assertLessEqual(sev.score, 15)

    def test_bridge_recommendations(self):
        items = [
            DamageItem(
                type="Rust",
                confidence=0.88,
                bounding_box=BoundingBox(x1=50, y1=50, x2=200, y2=200),
            )
        ]
        sev = calculate_severity(items, 800, 600, "bridge")
        rec = generate_recommendation("bridge", items, sev)
        self.assertIn("corrosion", rec.action.lower())


if __name__ == "__main__":
    unittest.main()
