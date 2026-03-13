from __future__ import annotations

import numpy as np


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    min_value = float(depth_map.min())
    max_value = float(depth_map.max())
    if max_value - min_value <= 1e-6:
        return np.zeros_like(depth_map, dtype=np.float32)
    return ((depth_map - min_value) / (max_value - min_value)).astype(np.float32)
