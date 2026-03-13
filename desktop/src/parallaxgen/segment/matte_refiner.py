from __future__ import annotations

import numpy as np


def refine_alpha(alpha: np.ndarray) -> np.ndarray:
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)
