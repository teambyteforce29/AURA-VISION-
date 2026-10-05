"""API endpoint tests using FastAPI TestClient."""

import os
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app


def test_api():
    client = TestClient(app)

    # 1. Health check
    print("Testing GET /api/health...")
    with client:
        resp = client.get("/api/health")
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        print("Health response:", data)
        assert data["status"] == "healthy"
        assert len(data["loaded_models"]) == 4

        # 2. Analyze road image
        sample_img_path = ROOT / "tests" / "sample_images" / "test_road.jpg"
        print(f"\nTesting POST /api/analyze (road) with {sample_img_path}...")
        with open(sample_img_path, "rb") as f:
            files = {"file": ("test_road.jpg", f, "image/jpeg")}
            data = {"asset_type": "road"}
            post_resp = client.post("/api/analyze", files=files, data=data)

        assert post_resp.status_code == 200, f"Analyze road failed: {post_resp.text}"
        road_res = post_resp.json()
        print("Road API response summary:")
        print("  Asset:", road_res["asset_type"])
        print("  Model:", road_res["model"])
        print("  Severity:", road_res["severity"])
        print("  Condition:", road_res["condition"])
        print("  Recommendation:", road_res["recommendation"])
        print("  Has annotated image base64:", bool(road_res.get("annotated_image")))
        assert road_res["asset_type"] == "road"
        assert road_res["model"]["name"] == "AURA_Vision_Road_Damage_YOLO11s"

        # 3. Analyze building image
        bld_img_path = ROOT / "tests" / "sample_images" / "test_building.jpg"
        print(f"\nTesting POST /api/analyze (building) with {bld_img_path}...")
        with open(bld_img_path, "rb") as f:
            files = {"file": ("test_building.jpg", f, "image/jpeg")}
            data = {"asset_type": "building"}
            post_resp = client.post("/api/analyze", files=files, data=data)

        assert post_resp.status_code == 200, f"Analyze building failed: {post_resp.text}"
        bld_res = post_resp.json()
        print("Building API response summary:")
        print("  Asset:", bld_res["asset_type"])
        print("  Model:", bld_res["model"])
        print("  Damage count:", len(bld_res["damage"]))
        if bld_res["damage"]:
            print("  Damage bbox is None:", bld_res["damage"][0]["bounding_box"] is None)
            assert bld_res["damage"][0]["bounding_box"] is None
        print("  Severity:", bld_res["severity"])
        print("  Condition:", bld_res["condition"])
        print("  Recommendation:", bld_res["recommendation"])

    print("\n=== ALL FASTAPI ENDPOINT TESTS PASSED ===")


if __name__ == "__main__":
    test_api()
