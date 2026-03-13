from __future__ import annotations

import json
from pathlib import Path

from parallaxgen.models import (
    PACKAGE_CONTRACT,
    CorpusIndexEntry,
    CorpusManifest,
    WallpaperPackage,
)


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
    wallpaper.validate_rendered_assets()
    (package_dir / PACKAGE_CONTRACT.meta_filename).write_text(
        json.dumps(wallpaper.meta.to_dict(), indent=2), encoding="utf-8"
    )

    for asset_name in wallpaper.required_asset_names():
        (package_dir / asset_name).write_bytes(wallpaper.rendered_assets[asset_name])

    # Write any extra assets (e.g. qa_grid.webp) that aren't in the contract
    # but were generated for development / QA purposes.
    extras = set(wallpaper.rendered_assets) - set(wallpaper.required_asset_names())
    for asset_name in sorted(extras):
        (package_dir / asset_name).write_bytes(wallpaper.rendered_assets[asset_name])
