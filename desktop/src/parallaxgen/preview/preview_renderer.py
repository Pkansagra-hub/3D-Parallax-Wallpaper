from __future__ import annotations

from pathlib import Path


def render_preview_summary(image_path: Path, include_clock: bool) -> dict[str, object]:
    return {
        "image": str(image_path),
        "mode": "clock_preview" if include_clock else "scene_preview",
        "suggested_clock_rect": [0.16, 0.07, 0.84, 0.30],
        "notes": [
            "Replace this summary with a real preview renderer once layer export is implemented.",
            "Use the same placement metadata in the Android preview surface.",
        ],
    }
