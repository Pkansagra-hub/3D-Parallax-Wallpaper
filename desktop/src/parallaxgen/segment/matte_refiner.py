from __future__ import annotations

import cv2
import numpy as np


def refine_alpha(
    alpha: np.ndarray,
    edge_radius: int = 3,
    smooth_sigma: float = 0.8,
) -> np.ndarray:
    """Clean raw segmentation alpha with morphological ops and edge smoothing.

    1. Morphological *open* to remove small noise islands outside the subject.
    2. Morphological *close* to fill small holes inside the subject.
    3. Gaussian edge smoothing to reduce staircase artifacts.
    """
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    mask_u8 = (alpha * 255).astype(np.uint8)

    ksize = edge_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # Remove noise islands
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    # Fill small holes
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Smooth edges
    if smooth_sigma > 0:
        blur_k = int(smooth_sigma * 6) | 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (blur_k, blur_k), smooth_sigma)

    return mask_u8.astype(np.float32) / 255.0
