"""
vision.py
---------
Intelligently handles both PDFs and raw Images (JPG, PNG) and converts
them into base64-encoded PNG strings for the Vision-Language Model.
"""

import base64
import io
import logging
from pathlib import Path

from pdf2image import convert_from_bytes
from PIL import Image

logger = logging.getLogger(__name__)

# Horizontal resolution in DPI for the rendered invoice image.
_DPI: int = 200


def pdf_to_base64_image(pdf_bytes: bytes, dpi: int = _DPI) -> str:
    """Convert the first page of a PDF into a base64-encoded JPEG string."""
    pages: list[Image.Image] = convert_from_bytes(pdf_bytes, dpi=dpi)

    if not pages:
        raise ValueError("The uploaded PDF contains no renderable pages.")

    first_page: Image.Image = pages[0]
    return _image_to_base64(first_page)


def raw_image_to_base64(image_bytes: bytes) -> str:
    """Process a native image (JPG/PNG) into a base64-encoded JPEG string."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image file: {exc}")
        
    return _image_to_base64(img)


def _image_to_base64(img: Image.Image, max_dim: int = 1600) -> str:
    """Helper: scales down large images and encodes as a compressed JPEG string."""
    # Convert to RGB (strips alpha channel, required for JPEG)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
        
    # Smart resize: if the image is massive, scale it down to reduce latency
    width, height = img.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    # Compress as JPEG to drastically reduce payload size (vs lossless PNG)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def process_file_to_base64(file_bytes: bytes, filename: str, content_type: str = "") -> str:
    """
    Route the file to the correct processor based on extension or content type.
    """
    filename_lower = filename.lower()
    content_type_lower = content_type.lower()

    if "pdf" in content_type_lower or filename_lower.endswith(".pdf"):
        logger.info("Processing as PDF...")
        return pdf_to_base64_image(file_bytes)
        
    elif content_type_lower.startswith("image/") or filename_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
        logger.info("Processing as native Image...")
        return raw_image_to_base64(file_bytes)
        
    else:
        raise ValueError(f"Unsupported file format. Must be PDF, JPG, or PNG. Received: {filename}")
