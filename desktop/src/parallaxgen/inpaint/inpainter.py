from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def inpaint_background(
    background_image: Image.Image,
    subject_alpha: np.ndarray,
    method: str = "auto",
) -> Image.Image:
    """Fill the hole left by the subject in the background layer.

    Parameters
    ----------
    background_image:
        The full source image (RGB or RGBA).
    subject_alpha:
        Refined subject matte, float32 ``[0, 1]``, shape ``(H, W)``.
    method:
        ``"auto"`` to prefer LaMa if installed, else cv2 (recommended).
        ``"lama"`` for ML inpainting (requires ``simple-lama-inpainting``).
        ``"cv2"`` for fast OpenCV Telea fallback.

    Returns
    -------
    Inpainted background with the subject region filled.
    """
    # Build binary inpaint mask — dilate slightly so edges are covered.
    mask_u8 = (subject_alpha > 0.25).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    bg_rgb = np.array(background_image.convert("RGB"))

    if method == "auto":
        try:
            import simple_lama_inpainting  # noqa: F401

            method = "lama"
        except ImportError:
            method = "cv2"

    if method == "lama":
        try:
            return _inpaint_lama(bg_rgb, mask_u8)
        except Exception:
            logger.warning("LaMa inpainting failed, falling back to cv2")
            return _inpaint_cv2(bg_rgb, mask_u8)

    if method == "cv2":
        return _inpaint_cv2(bg_rgb, mask_u8)

    logger.warning("Unknown inpaint method '%s', returning original", method)
    return background_image


def _inpaint_cv2(image_bgr: np.ndarray, mask: np.ndarray) -> Image.Image:
    """OpenCV Telea inpainting — fast, reasonable quality for backgrounds."""
    bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _inpaint_lama(image_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """LaMa ML inpainting via simple-lama-inpainting package."""
    from simple_lama_inpainting import SimpleLama  # type: ignore[import-untyped]

    lama = SimpleLama()
    img_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask)
    result = lama(img_pil, mask_pil)
    return result
