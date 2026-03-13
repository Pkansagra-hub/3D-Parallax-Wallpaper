from __future__ import annotations

import numpy as np


def build_clock_occlusion_mask(width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    center_band_top = int(height * 0.12)
    center_band_bottom = int(height * 0.45)
    left = int(width * 0.33)
    right = int(width * 0.67)
    mask[center_band_top:center_band_bottom, left:right] = 255
    return mask
