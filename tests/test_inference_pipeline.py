"""Integration test to verify loading and inference on all 4 model checkpoints."""

import os
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from backend.main import load_all_models
from backend.inference.router import process_inspection


def test_models_integration():
    print("=== Testing Model Loading ===")
    models = load_all_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    assert len(models) == 4, f"Expected 4 models, got {len(models)}"

    sample_dir = Path(__file__).parent / "sample_images"
    test_cases = [
        ("road", sample_dir / "test_road.jpg"),
        ("bridge", sample_dir / "test_bridge.jpg"),
        ("building", sample_dir / "test_building.jpg"),
        ("concrete", sample_dir / "test_bridge.jpg"),  # concrete surface
    ]

    for asset_type, img_path in test_cases:
        print(f"\n--- Testing Inference for '{asset_type}' using {img_path.name} ---")
        img = Image.open(img_path).convert("RGB")
        resp = process_inspection(
            asset_type=asset_type,
            pil_image=img,
            model_registry=models,
        )
        print(f"Asset: {resp.asset_type}")
        print(f"Model: {resp.model.name} ({resp.model.task})")
        print(f"Detected defects count: {len(resp.damage)}")
        for d in resp.damage[:3]:
            print(f"  - {d.type} (conf: {d.confidence:.2f}, bbox: {d.bounding_box})")
        print(f"Severity: {resp.severity.level} (Score: {resp.severity.score})")
        print(f"Condition: {resp.condition.score} / 100 ({resp.condition.status})")
        print(f"Recommendation: [{resp.recommendation.priority}] {resp.recommendation.action}")
        assert resp.model.name is not None
        assert resp.severity.level in ["Low", "Medium", "High", "Critical"]
        assert 0 <= resp.severity.score <= 100
        assert 0 <= resp.condition.score <= 100
        print("  -> PASSED")

    print("\n=== ALL INFERENCE INTEGRATION TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    test_models_integration()
