from pathlib import Path

from parallaxgen.compose.scene_builder import build_scene_package
from parallaxgen.corpus.manifest import build_manifest


def test_manifest_contains_generated_wallpaper(tmp_path: Path) -> None:
    image_path = tmp_path / "mountain.png"
    image_path.write_bytes(b"fake")

    wallpaper = build_scene_package(image_path)
    manifest = build_manifest([wallpaper])

    assert manifest.wallpapers[0].wallpaper_id == "mountain"
