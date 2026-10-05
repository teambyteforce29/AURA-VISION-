"""Generates synthetic test images for verifying the AURA Vision pipeline."""

import os
import numpy as np
from PIL import Image, ImageDraw


def create_test_images():
    output_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Road surface simulation (asphalt texture with simulated cracks)
    road_arr = np.random.normal(loc=110, scale=15, size=(640, 640, 3)).clip(0, 255).astype(np.uint8)
    road_img = Image.fromarray(road_arr)
    draw = ImageDraw.Draw(road_img)
    # Draw dark crack-like lines
    draw.line([(150, 200), (220, 260), (310, 290), (420, 380)], fill=(30, 30, 30), width=4)
    draw.line([(220, 260), (280, 230), (350, 240)], fill=(35, 35, 35), width=3)
    # Draw simulated pothole
    draw.ellipse([380, 150, 520, 280], fill=(25, 25, 25), outline=(15, 15, 15), width=3)
    road_path = os.path.join(output_dir, "test_road.jpg")
    road_img.save(road_path, quality=95)

    # 2. Bridge surface simulation (concrete/steel with rust patch)
    bridge_arr = np.random.normal(loc=160, scale=10, size=(640, 640, 3)).clip(0, 255).astype(np.uint8)
    bridge_img = Image.fromarray(bridge_arr)
    draw = ImageDraw.Draw(bridge_img)
    # Draw rust-colored patch
    draw.ellipse([180, 120, 440, 360], fill=(160, 65, 25))
    draw.line([(100, 450), (300, 480), (550, 500)], fill=(40, 40, 40), width=5)
    bridge_path = os.path.join(output_dir, "test_bridge.jpg")
    bridge_img.save(bridge_path, quality=95)

    # 3. Building wall simulation
    bld_arr = np.full((224, 224, 3), 210, dtype=np.uint8)
    bld_img = Image.fromarray(bld_arr)
    draw = ImageDraw.Draw(bld_img)
    draw.line([(50, 20), (90, 80), (130, 150), (170, 210)], fill=(40, 40, 40), width=2)
    bld_path = os.path.join(output_dir, "test_building.jpg")
    bld_img.save(bld_path, quality=95)

    print(f"Sample test images created in: {output_dir}")
    return road_path, bridge_path, bld_path


if __name__ == "__main__":
    create_test_images()
