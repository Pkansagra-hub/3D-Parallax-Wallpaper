from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SubjectMask:
    alpha: np.ndarray
    bbox: tuple[float, float, float, float]


class SubjectRunner:
    def infer(self, image_path: Path, width: int, height: int) -> SubjectMask:
        alpha = np.zeros((height, width), dtype=np.float32)
        left = int(width * 0.27)
        right = int(width * 0.73)
        top = int(height * 0.14)
        bottom = int(height * 0.87)
        alpha[top:bottom, left:right] = 1.0
        return SubjectMask(alpha=alpha, bbox=(0.27, 0.14, 0.73, 0.87))
