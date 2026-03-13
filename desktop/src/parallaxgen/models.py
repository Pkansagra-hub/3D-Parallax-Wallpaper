from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LayerSpec:
    name: str
    asset_path: str
    weight: float
    blur_px: float = 0.0


@dataclass(slots=True)
class WallpaperMeta:
    wallpaper_id: str
    version: int = 2
    resolution: tuple[int, int] = (1080, 2400)
    layer_count: int = 5
    clock_plane_index: int = 3
    parallax_strength: float = 0.65
    overscan: float = 0.18
    motion_profile: str = "cinematic_slow"
    depth_weights: list[float] = field(
        default_factory=lambda: [0.08, 0.18, 0.32, 0.48, 0.62]
    )
    clock_weight: float = 0.24
    blur_px: list[float] = field(default_factory=lambda: [1.2, 0.8, 0.3, 0.0, 0.0])
    safe_clock_rect: tuple[float, float, float, float] = (0.16, 0.07, 0.84, 0.30)
    focus_anchor: tuple[float, float] = (0.50, 0.36)
    subject_bbox: tuple[float, float, float, float] = (0.27, 0.14, 0.73, 0.87)
    has_clock_occlusion: bool = True
    inpainted: bool = True
    depth_model: str = "midas_dpt_large"
    segmentation_model: str = "birefnet_or_equivalent"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["id"] = payload.pop("wallpaper_id")
        return payload


@dataclass(slots=True)
class WallpaperPackage:
    wallpaper_id: str
    title: str
    meta: WallpaperMeta
    layers: list[LayerSpec]
    preview_asset: str


@dataclass(slots=True)
class CorpusIndexEntry:
    wallpaper_id: str
    title: str
    preview: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.wallpaper_id, "title": self.title, "preview": self.preview}


@dataclass(slots=True)
class CorpusManifest:
    wallpapers: list[CorpusIndexEntry]
    format: str = "parallaxgen-corpus-v2"

    def write(self, path: Path) -> None:
        payload = {
            "format": self.format,
            "wallpapers": [entry.to_dict() for entry in self.wallpapers],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
