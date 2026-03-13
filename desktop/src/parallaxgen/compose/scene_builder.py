from __future__ import annotations

from pathlib import Path

from parallaxgen.compose.layer_planner import plan_layers
from parallaxgen.config import PipelineConfig
from parallaxgen.depth.depth_runner import DepthRunner
from parallaxgen.models import PACKAGE_CONTRACT, WallpaperMeta, WallpaperPackage
from parallaxgen.segment.matte_refiner import refine_alpha
from parallaxgen.segment.subject_runner import SubjectRunner


def build_scene_package(
    image_path: Path,
    title: str | None = None,
    config: PipelineConfig | None = None,
) -> WallpaperPackage:
    config = config or PipelineConfig()
    wallpaper_id = image_path.stem.lower().replace(" ", "_")
    depth_result = DepthRunner(
        model_name=config.depth_model,
        output_resolution=config.output_resolution,
    ).infer(image_path)
    subject_mask = SubjectRunner().infer(
        image_path, depth_result.width, depth_result.height
    )
    refined_alpha = refine_alpha(subject_mask.alpha)
    planned_scene = plan_layers(wallpaper_id, config=config)

    meta = WallpaperMeta(
        wallpaper_id=wallpaper_id,
        target_device=config.target_device,
        resolution=(depth_result.width, depth_result.height),
        parallax_strength=config.parallax_strength,
        overscan=config.overscan,
        motion_profile=config.motion_profile,
        depth_weights=config.depth_weights.copy(),
        blur_px=config.blur_px.copy(),
        safe_clock_rect=planned_scene.safe_clock_rect,
        subject_bbox=subject_mask.bbox,
        depth_model=config.depth_model,
        segmentation_model=config.segmentation_model,
        inpainted=refined_alpha.any().item(),
    )

    return WallpaperPackage(
        wallpaper_id=wallpaper_id,
        title=title or image_path.stem.replace("_", " ").title(),
        meta=meta,
        layers=planned_scene.layers,
        preview_asset=f"{wallpaper_id}/{PACKAGE_CONTRACT.required_support_assets[-1]}",
    )
