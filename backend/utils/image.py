"""Image loading, format validation, and decoding utilities."""

import base64
import io
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 32


def validate_image_bytes(image_bytes: bytes) -> Image.Image:
    """Validates that bytes represent a readable, uncorrupted image of supported format.

    Args:
        image_bytes: Raw binary bytes of the uploaded file.

    Returns:
        PIL.Image.Image: The validated and decoded RGB PIL Image.

    Raises:
        ValueError: If file is empty, corrupted, or unsupported.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Uploaded image file is empty.")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()  # Verify file header and integrity
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image data: {str(e)}")

    # Re-open after verify() since verify() destroys internal image pointers
    pil_image = Image.open(io.BytesIO(image_bytes))

    format_name = (pil_image.format or "").upper()
    if format_name not in SUPPORTED_FORMATS and format_name != "JPEG":
        raise ValueError(
            f"Unsupported image format: '{format_name}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Correct orientation if EXIF orientation tag is present
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass

    # Convert to RGB mode (handles RGBA, Palette, Grayscale, etc.)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    width, height = pil_image.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ValueError(f"Image dimensions too small ({width}x{height}). Minimum is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}.")

    return pil_image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL Image in RGB to an OpenCV BGR numpy array."""
    rgb_arr = np.array(pil_image)
    # RGB to BGR
    return rgb_arr[:, :, ::-1].copy()


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR numpy array to a PIL Image in RGB."""
    rgb_arr = cv2_image[:, :, ::-1]
    return Image.fromarray(rgb_arr)


def image_to_base64(image: Union[Image.Image, np.ndarray], format: str = "JPEG") -> str:
    """Encodes a PIL Image or OpenCV array into a Base64 data string."""
    if isinstance(image, np.ndarray):
        pil_img = cv2_to_pil(image)
    else:
        pil_img = image

    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def base64_to_pil(base64_str: str) -> Image.Image:
    """Decodes a Base64 data URI or raw base64 string to a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    decoded = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(decoded)).convert("RGB")
