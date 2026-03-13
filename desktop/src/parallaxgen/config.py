from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from parallaxgen.models import (
    DEFAULT_DEPTH_WEIGHTS,
    DEFAULT_LAYER_BLUR,
    TARGET_DEVICE_NAME,
    TARGET_RENDER_RESOLUTION,
)


@dataclass(slots=True)
class QualityThresholds:
    min_depth_separation: float = 0.18
    min_subject_coverage: float = 0.08
    min_clock_clearance: float = 0.55


@dataclass(slots=True)
class PipelineConfig:
    target_device: str = TARGET_DEVICE_NAME
    output_resolution: tuple[int, int] = TARGET_RENDER_RESOLUTION
    depth_model: str = "depth_anything_v2_large"
    segmentation_model: str = "birefnet"
    overscan: float = 0.18
    parallax_strength: float = 0.65
    motion_profile: str = "cinematic_slow"
    safe_clock_rect: tuple[float, float, float, float] = (0.16, 0.07, 0.84, 0.30)
    depth_weights: list[float] = field(
        default_factory=lambda: DEFAULT_DEPTH_WEIGHTS.copy()
    )
    blur_px: list[float] = field(default_factory=lambda: DEFAULT_LAYER_BLUR.copy())
    max_blur_px: float = 6.0
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def write_json(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")


def parse_clock_safe_rect(value: str | None) -> tuple[float, float, float, float]:
    if not value:
        return (0.16, 0.07, 0.84, 0.30)

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Clock safe rect must have four comma-separated floats.")

    left, top, right, bottom = (float(part) for part in parts)
    return (left, top, right, bottom)


def build_pipeline_config(
    *,
    width: int,
    height: int,
    depth_model: str,
    segmentation_model: str,
    overscan: float,
    parallax_strength: float,
    motion_profile: str,
    clock_safe_rect: str | None,
    min_depth_separation: float,
    min_subject_coverage: float,
    min_clock_clearance: float,
) -> PipelineConfig:
    return PipelineConfig(
        output_resolution=(width, height),
        depth_model=depth_model,
        segmentation_model=segmentation_model,
        overscan=overscan,
        parallax_strength=parallax_strength,
        motion_profile=motion_profile,
        safe_clock_rect=parse_clock_safe_rect(clock_safe_rect),
        quality_thresholds=QualityThresholds(
            min_depth_separation=min_depth_separation,
            min_subject_coverage=min_subject_coverage,
            min_clock_clearance=min_clock_clearance,
        ),
    )
