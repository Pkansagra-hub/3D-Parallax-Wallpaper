from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class DepthResult:
    width: int
    height: int
    depth_map: np.ndarray
    model_name: str = "midas_dpt_large"


class DepthRunner:
    def __init__(
        self,
        model_name: str = "midas_dpt_large",
        output_resolution: tuple[int, int] = (1440, 3120),
    ) -> None:
        self.model_name = model_name
        self.output_resolution = output_resolution

    def infer(self, image_path: Path) -> DepthResult:
        # Starter implementation returns a simple vertical ramp so the rest of the
        # package pipeline can be wired before the real model is integrated.
        width, height = self.output_resolution
        depth_map = np.tile(
            np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], (1, width)
        )
        return DepthResult(
            width=width, height=height, depth_map=depth_map, model_name=self.model_name
        )
