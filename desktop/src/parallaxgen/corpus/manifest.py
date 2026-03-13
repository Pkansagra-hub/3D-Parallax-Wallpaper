from __future__ import annotations

import json
from pathlib import Path

from parallaxgen.models import CorpusIndexEntry, CorpusManifest, WallpaperPackage


def build_manifest(wallpapers: list[WallpaperPackage]) -> CorpusManifest:
    entries = [
        CorpusIndexEntry(
            wallpaper_id=wallpaper.wallpaper_id,
            title=wallpaper.title,
            preview=wallpaper.preview_asset,
        )
        for wallpaper in wallpapers
    ]
    return CorpusManifest(wallpapers=entries)


def write_package_files(output_dir: Path, wallpaper: WallpaperPackage) -> None:
    package_dir = output_dir / wallpaper.wallpaper_id
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "meta.json").write_text(
        json.dumps(wallpaper.meta.to_dict(), indent=2), encoding="utf-8"
    )
    for layer in wallpaper.layers:
        (package_dir / f"{layer.name}.webp").write_bytes(b"")
    (package_dir / "clock_occlusion_mask.webp").write_bytes(b"")
    (package_dir / "subject_mask.webp").write_bytes(b"")
    (package_dir / "depth_map.webp").write_bytes(b"")
    (package_dir / "preview.webp").write_bytes(b"")
