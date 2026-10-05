"""Test suite for InspectionMetadata and Multi-Format Reports (PDF, TXT, JSON)."""

import os
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas.results import (
    AnalysisResponse,
    AnalysisResult,
    BoundingBox,
    ConditionResult,
    DamageItem,
    InspectionMetadata,
    ModelInfo,
    RecommendationResult,
    SeverityResult,
)
from backend.services.reports import (
    generate_json_report,
    generate_pdf_report,
    generate_text_report,
)
from fastapi.testclient import TestClient
from backend.main import app


def test_schema_with_metadata():
    print("=== Testing InspectionMetadata Schema ===")
    meta = InspectionMetadata(
        asset_name="Golden Gate Approach Pier 4",
        asset_type="bridge",
        location="GPS: 37.8199° N, 122.4783° W",
        inspector_name="Dr. Sarah Connor, PE",
        timestamp="2026-10-05 14:30:00",
    )
    result = AnalysisResult(
        inspection_metadata=meta,
        asset_type="bridge",
        model=ModelInfo(name="AURA_Vision_Bridge_Damage_YOLO11s", task="object_detection"),
        damage=[
            DamageItem(
                type="Rust",
                confidence=0.89,
                bounding_box=BoundingBox(x1=100.0, y1=150.0, x2=320.0, y2=400.0),
            ),
            DamageItem(
                type="Spalling",
                confidence=0.78,
                bounding_box=BoundingBox(x1=340.0, y1=200.0, x2=450.0, y2=310.0),
            ),
        ],
        severity=SeverityResult(level="High", score=72),
        condition=ConditionResult(score=48, status="Needs Maintenance"),
        recommendation=RecommendationResult(
            action="Remove corrosion, assess structural section loss, apply anti-corrosion coating.",
            priority="High",
        ),
    )
    assert result.inspection_metadata.asset_name == "Golden Gate Approach Pier 4"
    assert result.inspection_metadata.inspector_name == "Dr. Sarah Connor, PE"
    print("InspectionMetadata Schema test PASSED")
    return result.model_dump()


def test_report_generation(data_dict):
    print("\n=== Testing Multi-Format Report Generation ===")
    # 1. Text report
    txt = generate_text_report(data_dict)
    assert "Golden Gate Approach Pier 4" in txt
    assert "Dr. Sarah Connor, PE" in txt
    assert "Rust (89%)" in txt
    print("  -> Plaintext Report generated successfully (length:", len(txt), "chars)")

    # 2. PDF report
    pdf_bytes = generate_pdf_report(data_dict)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")
    print("  -> PDF Report generated successfully (size:", len(pdf_bytes), "bytes, valid %PDF header)")

    # 3. JSON report
    json_str = generate_json_report(data_dict)
    assert "Golden Gate Approach Pier 4" in json_str
    print("  -> JSON Report generated successfully")

    print("Multi-Format Report Generation test PASSED")


def test_api_with_metadata():
    print("\n=== Testing API Endpoint with Form Metadata ===")
    client = TestClient(app)
    sample_img_path = ROOT / "tests" / "sample_images" / "test_bridge.jpg"

    with client:
        with open(sample_img_path, "rb") as f:
            files = {"file": ("test_bridge.jpg", f, "image/jpeg")}
            data = {
                "asset_type": "bridge",
                "asset_name": "Overpass Viaduct #21",
                "location": "Northbound Mile 48.2",
                "inspector_name": "Marcus Vance, Structural Inspector",
            }
            resp = client.post("/api/analyze", files=files, data=data)

        assert resp.status_code == 200, f"API failed: {resp.text}"
        payload = resp.json()
        print("API Response Metadata:", payload.get("inspection_metadata"))
        meta = payload.get("inspection_metadata")
        assert meta is not None
        assert meta["asset_name"] == "Overpass Viaduct #21"
        assert meta["location"] == "Northbound Mile 48.2"
        assert meta["inspector_name"] == "Marcus Vance, Structural Inspector"

        # Verify PDF report generation on API payload
        pdf_bytes = generate_pdf_report(payload)
        assert pdf_bytes.startswith(b"%PDF")
        print("PDF generated from API response: valid (%PDF header confirmed)")

    print("API Metadata integration test PASSED")


if __name__ == "__main__":
    data = test_schema_with_metadata()
    test_report_generation(data)
    test_api_with_metadata()
    print("\n=== ALL DELTA PROMPT TESTS PASSED SUCCESSFULLY ===")
