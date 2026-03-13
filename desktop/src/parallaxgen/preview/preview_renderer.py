from __future__ import annotations

from pathlib import Path

from parallaxgen.config import PipelineConfig


def render_preview_summary(
    image_path: Path,
    include_clock: bool,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    config = config or PipelineConfig()
    return {
        "image": str(image_path),
        "mode": "clock_preview" if include_clock else "scene_preview",
        "suggested_clock_rect": list(config.safe_clock_rect),
        "config": config.to_dict(),
        "notes": [
            "Replace this summary with a real preview renderer once layer export is implemented.",
            "Use the same placement metadata in the Android preview surface.",
        ],
    }
