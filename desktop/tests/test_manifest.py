import json
from pathlib import Path

from parallaxgen.compose.scene_builder import build_scene_package
from parallaxgen.config import PipelineConfig, build_pipeline_config
from parallaxgen.corpus.manifest import build_manifest, write_package_files
from parallaxgen.models import (
    PACKAGE_CONTRACT,
    TARGET_DEVICE_NAME,
    TARGET_RENDER_RESOLUTION,
)
from PIL import Image


def _test_config() -> PipelineConfig:
    """Config that uses synthetic backends to avoid model downloads."""
    return PipelineConfig(depth_model="synthetic", segmentation_model="synthetic")


def test_manifest_contains_generated_wallpaper(tmp_path: Path) -> None:
    image_path = tmp_path / "mountain.png"
    Image.new("RGB", (900, 1600), color=(60, 120, 200)).save(image_path)

    wallpaper = build_scene_package(image_path, config=_test_config())
    manifest = build_manifest([wallpaper])

    assert manifest.wallpapers[0].wallpaper_id == "mountain"


def test_scene_package_uses_s26_ultra_defaults(tmp_path: Path) -> None:
    image_path = tmp_path / "lagoon.png"
    Image.new("RGB", (900, 1600), color=(10, 80, 180)).save(image_path)

    wallpaper = build_scene_package(image_path, config=_test_config())

    assert wallpaper.meta.target_device == TARGET_DEVICE_NAME
    assert wallpaper.meta.resolution == TARGET_RENDER_RESOLUTION


def test_package_writer_uses_contract_asset_names(tmp_path: Path) -> None:
    image_path = tmp_path / "storm.png"
    Image.new("RGB", (900, 1600), color=(15, 25, 60)).save(image_path)

    wallpaper = build_scene_package(image_path, config=_test_config())
    output_dir = tmp_path / "corpus"

    write_package_files(output_dir, wallpaper)

    package_dir = output_dir / wallpaper.wallpaper_id
    meta_payload = json.loads(
        (package_dir / PACKAGE_CONTRACT.meta_filename).read_text(encoding="utf-8")
    )

    assert meta_payload["version"] == 2
    assert meta_payload["id"] == wallpaper.wallpaper_id
    for asset_name in wallpaper.required_asset_names():
        asset_path = package_dir / asset_name
        assert asset_path.exists()
        assert asset_path.stat().st_size > 0

    with Image.open(package_dir / "preview.webp") as preview:
        assert preview.size == TARGET_RENDER_RESOLUTION

    with Image.open(package_dir / "depth_map.webp") as depth_map:
        assert depth_map.size == TARGET_RENDER_RESOLUTION


def test_pipeline_config_serializes_for_debugging() -> None:
    config = PipelineConfig()

    payload = config.to_dict()

    assert payload["target_device"] == TARGET_DEVICE_NAME
    assert payload["output_resolution"] == TARGET_RENDER_RESOLUTION
    assert payload["quality_thresholds"]["min_clock_clearance"] == 0.55


def test_scene_package_respects_config_overrides(tmp_path: Path) -> None:
    image_path = tmp_path / "shoreline.png"
    Image.new("RGB", (900, 1600), color=(180, 120, 60)).save(image_path)
    config = build_pipeline_config(
        width=1080,
        height=2400,
        depth_model="synthetic",
        segmentation_model="synthetic",
        overscan=0.22,
        parallax_strength=0.72,
        motion_profile="studio_preview",
        clock_safe_rect="0.10,0.05,0.90,0.28",
        min_depth_separation=0.2,
        min_subject_coverage=0.1,
        min_clock_clearance=0.6,
    )

    wallpaper = build_scene_package(image_path=image_path, config=config)

    assert wallpaper.meta.resolution == (1080, 2400)
    assert wallpaper.meta.depth_model == "synthetic"
    assert wallpaper.meta.segmentation_model == "synthetic"
    assert wallpaper.meta.overscan == 0.22
    assert wallpaper.meta.parallax_strength == 0.72
    assert wallpaper.meta.motion_profile == "studio_preview"
    # safe_clock_rect may be recomputed by compute_safe_clock_rect; verify it
    # is a valid normalised rectangle rather than a fixed value.
    l, t, r, b = wallpaper.meta.safe_clock_rect
    assert 0.0 <= l < r <= 1.0 and 0.0 <= t < b <= 1.0


def test_scene_package_contains_non_empty_rendered_assets(tmp_path: Path) -> None:
    image_path = tmp_path / "island.png"
    Image.new("RGB", (900, 1600), color=(20, 120, 210)).save(image_path)

    wallpaper = build_scene_package(image_path=image_path, config=_test_config())

    assert set(wallpaper.required_asset_names()).issubset(
        set(wallpaper.rendered_assets)
    )
    assert all(
        wallpaper.rendered_assets[name] for name in wallpaper.required_asset_names()
    )
