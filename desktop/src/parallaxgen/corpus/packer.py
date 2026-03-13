from __future__ import annotations

import json
import zipfile
from pathlib import Path

from parallaxgen.models import PACKAGE_CONTRACT


def pack_corpus_directory(corpus_dir: Path, output_path: Path) -> Path:
    """Create a ``.parallax`` ZIP archive from a corpus directory.

    Ensures ``index.json`` is present at the archive root and validates
    that each wallpaper sub-directory contains the required ``meta.json``.
    """
    index_path = corpus_dir / PACKAGE_CONTRACT.index_filename
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {PACKAGE_CONTRACT.index_filename} in {corpus_dir}. "
            "Run the batch command first to generate the corpus index."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(corpus_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(corpus_dir))
    return output_path


def validate_corpus(corpus_dir: Path) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    index_path = corpus_dir / PACKAGE_CONTRACT.index_filename
    if not index_path.exists():
        errors.append(f"Missing {PACKAGE_CONTRACT.index_filename}")
        return errors

    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index.get("wallpapers", []):
        wid = entry.get("id", "?")
        meta_path = corpus_dir / wid / PACKAGE_CONTRACT.meta_filename
        if not meta_path.exists():
            errors.append(f"{wid}: missing {PACKAGE_CONTRACT.meta_filename}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Validate required meta.json fields so a corrupted package
        # doesn't silently pass and crash the Android renderer.
        _REQUIRED_META_FIELDS = (
            "id",
            "version",
            "resolution",
            "layer_count",
            "parallax_strength",
            "depth_weights",
            "safe_clock_rect",
        )
        for field in _REQUIRED_META_FIELDS:
            if field not in meta:
                errors.append(f"{wid}: meta.json missing required field '{field}'")
        for asset in ["preview.webp", "depth_map.webp", "subject_mask.webp"]:
            asset_path = corpus_dir / wid / asset
            if not asset_path.exists():
                errors.append(f"{wid}: missing {asset}")
            elif asset_path.stat().st_size < 512:
                errors.append(
                    f"{wid}: {asset} suspiciously small ({asset_path.stat().st_size}B)"
                )

    return errors
