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


def test_manifest_contains_generated_wallpaper(tmp_path: Path) -> None:
    image_path = tmp_path / "mountain.png"
    image_path.write_bytes(b"fake")

    wallpaper = build_scene_package(image_path)
    manifest = build_manifest([wallpaper])

    assert manifest.wallpapers[0].wallpaper_id == "mountain"


def test_scene_package_uses_s26_ultra_defaults(tmp_path: Path) -> None:
    image_path = tmp_path / "lagoon.png"
    image_path.write_bytes(b"fake")

    wallpaper = build_scene_package(image_path)

    assert wallpaper.meta.target_device == TARGET_DEVICE_NAME
    assert wallpaper.meta.resolution == TARGET_RENDER_RESOLUTION


def test_package_writer_uses_contract_asset_names(tmp_path: Path) -> None:
    image_path = tmp_path / "storm.png"
    image_path.write_bytes(b"fake")

    wallpaper = build_scene_package(image_path)
    output_dir = tmp_path / "corpus"

    write_package_files(output_dir, wallpaper)

    package_dir = output_dir / wallpaper.wallpaper_id
    meta_payload = json.loads(
        (package_dir / PACKAGE_CONTRACT.meta_filename).read_text(encoding="utf-8")
    )

    assert meta_payload["version"] == 2
    assert meta_payload["id"] == wallpaper.wallpaper_id
    for asset_name in wallpaper.required_asset_names():
        assert (package_dir / asset_name).exists()


def test_pipeline_config_serializes_for_debugging() -> None:
    config = PipelineConfig()

    payload = config.to_dict()

    assert payload["target_device"] == TARGET_DEVICE_NAME
    assert payload["output_resolution"] == TARGET_RENDER_RESOLUTION
    assert payload["quality_thresholds"]["min_clock_clearance"] == 0.55


def test_scene_package_respects_config_overrides(tmp_path: Path) -> None:
    image_path = tmp_path / "shoreline.png"
    image_path.write_bytes(b"fake")
    config = build_pipeline_config(
        width=1080,
        height=2400,
        depth_model="custom_depth_model",
        segmentation_model="custom_segmenter",
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
    assert wallpaper.meta.depth_model == "custom_depth_model"
    assert wallpaper.meta.segmentation_model == "custom_segmenter"
    assert wallpaper.meta.overscan == 0.22
    assert wallpaper.meta.parallax_strength == 0.72
    assert wallpaper.meta.motion_profile == "studio_preview"
    assert wallpaper.meta.safe_clock_rect == (0.10, 0.05, 0.90, 0.28)
