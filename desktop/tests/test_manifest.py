import json
from pathlib import Path

import numpy as np
from parallaxgen.compose.layer_planner import (
    SceneType,
    _find_depth_centres,
    classify_scene,
    plan_layers,
)
from parallaxgen.compose.occlusion_planner import (
    compute_safe_clock_rect,
    derive_clock_layout,
)
from parallaxgen.compose.quality_scorer import score_scene
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


def test_clock_layout_is_derived_from_safe_rect() -> None:
    rect = (0.30, 0.16, 0.70, 0.28)

    anchor, font_scale = derive_clock_layout(rect)

    assert 0.30 < anchor[0] < 0.70
    assert 0.16 < anchor[1] < 0.28
    assert 0.28 <= font_scale <= 0.56


def test_clock_rect_prefers_compact_clear_window() -> None:
    h, w = 200, 120
    subject_alpha = np.zeros((h, w), dtype=np.float32)
    # Block left and right sides in the upper half, leaving a clear centre gap.
    subject_alpha[:90, :28] = 1.0
    subject_alpha[:90, 92:] = 1.0
    depth_map = np.tile(np.linspace(0.2, 0.95, h, dtype=np.float32)[:, None], (1, w))

    rect = compute_safe_clock_rect(
        subject_alpha,
        preferred_rect=(0.16, 0.07, 0.84, 0.30),
        depth_map=depth_map,
        scene_type="vista",
    )

    l, t, r, b = rect
    assert 0.25 <= l <= 0.35
    assert 0.65 <= r <= 0.75
    assert t < 0.35
    assert (r - l) < 0.55


def test_layer_planner_maps_far_band_to_far_depths() -> None:
    config = _test_config()
    depth_map = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None], (1, 48))
    subject_alpha = np.zeros_like(depth_map)

    scene = plan_layers(
        "gradient_scene",
        config=config,
        depth_map=depth_map,
        subject_alpha=subject_alpha,
    )

    far_mask = scene.layer_masks["layer_0_far_bg"]
    near_mask = scene.layer_masks["layer_2_near_mid"]

    assert float(far_mask[-1].mean()) > float(far_mask[0].mean())
    assert float(near_mask[0].mean()) > float(near_mask[-1].mean())


def test_quality_report_includes_plane_cohesion_metrics() -> None:
    depth_map = np.tile(np.linspace(0.0, 1.0, 80, dtype=np.float32)[:, None], (1, 60))
    subject_alpha = np.zeros_like(depth_map)
    subject_alpha[18:68, 18:42] = 1.0
    safe_rect = (0.16, 0.07, 0.84, 0.30)
    config = _test_config()
    scene = plan_layers(
        "quality_scene",
        config=config,
        depth_map=depth_map,
        subject_alpha=subject_alpha,
    )

    report = score_scene(
        depth_map,
        subject_alpha,
        safe_rect,
        config.quality_thresholds,
        layer_masks=scene.layer_masks,
    )

    payload = report.to_dict()

    assert "plane_cohesion" in payload
    assert "foreground_cohesion" in payload
    assert "layer_differentiation" in payload
    assert "scene_type" in payload
    assert 0.0 <= payload["plane_cohesion"] <= 1.0
    assert 0.0 <= payload["foreground_cohesion"] <= 1.0
    assert 0.0 <= payload["layer_differentiation"] <= 1.0


# ---------------------------------------------------------------------------
# Scene classification tests
# ---------------------------------------------------------------------------


def test_classify_scene_compact_subject_is_portrait() -> None:
    """A small compact subject with consistent depth → PORTRAIT."""
    depth_map = np.full((100, 80), 0.5, dtype=np.float32)
    # Place a compact square "subject" with tight depth range.
    subject_alpha = np.zeros((100, 80), dtype=np.float32)
    subject_alpha[30:70, 20:60] = 1.0
    depth_map[30:70, 20:60] = 0.25  # consistent depth

    result = classify_scene(subject_alpha, depth_map)
    assert result == SceneType.PORTRAIT


def test_classify_scene_terrain_with_depth_variance_is_vista() -> None:
    """A large mask with high depth variance → VISTA."""
    depth_map = np.tile(np.linspace(0.0, 1.0, 100, dtype=np.float32)[:, None], (1, 80))
    # Subject covers 40% of pixels but spans the entire depth range.
    subject_alpha = np.zeros((100, 80), dtype=np.float32)
    subject_alpha[10:90, 20:60] = 1.0  # 40% coverage

    result = classify_scene(subject_alpha, depth_map)
    assert result == SceneType.VISTA


def test_classify_scene_landscape_flag_forces_vista() -> None:
    """When SubjectRunner flags landscape, always VISTA regardless of mask."""
    depth_map = np.full((100, 80), 0.5, dtype=np.float32)
    subject_alpha = np.zeros((100, 80), dtype=np.float32)
    subject_alpha[30:70, 20:60] = 1.0

    result = classify_scene(subject_alpha, depth_map, is_landscape=True)
    assert result == SceneType.VISTA


def test_classify_scene_low_coverage_is_vista() -> None:
    """Coverage below 3% → VISTA (no meaningful subject)."""
    depth_map = np.full((200, 200), 0.5, dtype=np.float32)
    subject_alpha = np.zeros((200, 200), dtype=np.float32)
    subject_alpha[0:3, 0:3] = 1.0  # ~0.02%

    result = classify_scene(subject_alpha, depth_map)
    assert result == SceneType.VISTA


def test_classify_scene_edge_touching_is_vista() -> None:
    """Subject touching 3+ edges → VISTA (scene-filling geometry)."""
    depth_map = np.full((100, 80), 0.5, dtype=np.float32)
    subject_alpha = np.zeros((100, 80), dtype=np.float32)
    # Touches top, left, and bottom but not right (still 3 edges).
    subject_alpha[0:100, 0:30] = 1.0
    # Consistent depth so it wouldn't trip the variance check.
    depth_map[0:100, 0:30] = 0.3

    result = classify_scene(subject_alpha, depth_map)
    assert result == SceneType.VISTA


def test_vista_plan_has_four_distinct_layers() -> None:
    """VISTA decomposition should produce 4 layers with distinct alpha coverage."""
    config = _test_config()
    depth_map = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32)[:, None], (1, 96))
    subject_alpha = np.zeros_like(depth_map)

    scene = plan_layers(
        "gradient_vista",
        config=config,
        depth_map=depth_map,
        subject_alpha=subject_alpha,
    )

    assert scene.scene_type == SceneType.VISTA
    # Each active layer should have >5% coverage.
    for name in (
        "layer_0_far_bg",
        "layer_1_deep_mid",
        "layer_2_near_mid",
        "layer_3_hero_fg",
    ):
        mass = float((scene.layer_masks[name] > 0.15).mean())
        assert mass > 0.05, f"{name} has only {mass:.1%} coverage"
    # Front FX should be empty in VISTA mode.
    assert float(scene.layer_masks["layer_4_front_fx"].sum()) == 0.0


def test_quality_report_detects_redundant_layers() -> None:
    """When two layers have >85% IoU, the scorer should flag redundancy."""
    depth_map = np.full((80, 60), 0.5, dtype=np.float32)
    subject_alpha = np.zeros_like(depth_map)
    safe_rect = (0.16, 0.07, 0.84, 0.30)
    # Two layers with near-identical coverage.
    identical = np.ones_like(depth_map)
    masks = {
        "layer_0_far_bg": identical,
        "layer_1_deep_mid": identical * 0.95,
        "layer_2_near_mid": np.zeros_like(depth_map),
        "layer_3_hero_fg": np.zeros_like(depth_map),
        "layer_4_front_fx": np.zeros_like(depth_map),
    }

    report = score_scene(depth_map, subject_alpha, safe_rect, layer_masks=masks)
    assert any("Redundant" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Depth centre clustering tests
# ---------------------------------------------------------------------------


def test_depth_centres_find_bimodal_clusters() -> None:
    """K-means should place centres in both modes of a bimodal distribution."""
    rng = np.random.default_rng(42)
    sky = rng.normal(0.85, 0.03, 3000).clip(0.01, 0.99)
    ground = rng.normal(0.25, 0.08, 7000).clip(0.01, 0.99)
    flat = np.concatenate([sky, ground]).astype(np.float32)

    centres = _find_depth_centres(flat, n=4)

    assert len(centres) == 4
    # Sorted descending (far → near).
    assert centres == sorted(centres, reverse=True)
    # At least one centre in the sky zone and one in the ground zone.
    assert any(c > 0.7 for c in centres), f"No centre in sky zone: {centres}"
    assert any(c < 0.4 for c in centres), f"No centre in ground zone: {centres}"


def test_depth_centres_continuous_gradient() -> None:
    """Continuous depth should produce evenly spread centres."""
    flat = np.linspace(0.05, 0.95, 10000).astype(np.float32)

    centres = _find_depth_centres(flat, n=4)

    assert len(centres) == 4
    assert all(0.0 <= c <= 1.0 for c in centres), f"Out-of-range centres: {centres}"
    # Adjacent centres should be reasonably separated.
    for i in range(1, 4):
        gap = centres[i - 1] - centres[i]
        assert gap > 0.05, f"Centres too close: {centres}"


def test_bimodal_vista_all_bands_have_content() -> None:
    """Bimodal depth (sky 30% + ground 70%) should produce 4 populated bands."""
    config = _test_config()
    h, w = 128, 96
    depth_map = np.zeros((h, w), dtype=np.float32)
    # Sky: top 30% of image → far depth (~0.85)
    sky_rows = int(h * 0.30)
    depth_map[:sky_rows, :] = (
        np.random.default_rng(7)
        .normal(0.85, 0.03, (sky_rows, w))
        .clip(0.01, 0.99)
        .astype(np.float32)
    )
    # Ground: bottom 70% → near depth (~0.20)
    depth_map[sky_rows:, :] = (
        np.random.default_rng(8)
        .normal(0.20, 0.06, (h - sky_rows, w))
        .clip(0.01, 0.99)
        .astype(np.float32)
    )

    subject_alpha = np.zeros_like(depth_map)
    scene = plan_layers(
        "bimodal_test",
        config=config,
        depth_map=depth_map,
        subject_alpha=subject_alpha,
    )
    assert scene.scene_type == SceneType.VISTA
    # Every active layer should have ≥5% visual mass.
    for name in (
        "layer_0_far_bg",
        "layer_1_deep_mid",
        "layer_2_near_mid",
        "layer_3_hero_fg",
    ):
        mass = float((scene.layer_masks[name] > 0.15).mean())
        assert mass >= 0.05, f"{name} only {mass:.1%} in bimodal scene"
