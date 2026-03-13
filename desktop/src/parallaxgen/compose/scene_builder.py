from __future__ import annotations

from pathlib import Path

from parallaxgen.compose.layer_planner import plan_layers
from parallaxgen.depth.depth_runner import DepthRunner
from parallaxgen.models import WallpaperMeta, WallpaperPackage
from parallaxgen.segment.matte_refiner import refine_alpha
from parallaxgen.segment.subject_runner import SubjectRunner


def build_scene_package(image_path: Path, title: str | None = None) -> WallpaperPackage:
    wallpaper_id = image_path.stem.lower().replace(" ", "_")
    depth_result = DepthRunner().infer(image_path)
    subject_mask = SubjectRunner().infer(
        image_path, depth_result.width, depth_result.height
    )
    refined_alpha = refine_alpha(subject_mask.alpha)
    planned_scene = plan_layers(wallpaper_id)

    meta = WallpaperMeta(
        wallpaper_id=wallpaper_id,
        resolution=(depth_result.width, depth_result.height),
        safe_clock_rect=planned_scene.safe_clock_rect,
        subject_bbox=subject_mask.bbox,
        depth_model=depth_result.model_name,
        inpainted=refined_alpha.any().item(),
    )

    return WallpaperPackage(
        wallpaper_id=wallpaper_id,
        title=title or image_path.stem.replace("_", " ").title(),
        meta=meta,
        layers=planned_scene.layers,
        preview_asset=f"{wallpaper_id}/preview.webp",
    )
