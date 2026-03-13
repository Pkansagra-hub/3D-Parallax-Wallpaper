from __future__ import annotations

from dataclasses import dataclass

from parallaxgen.config import PipelineConfig
from parallaxgen.models import LayerSpec


@dataclass(slots=True)
class PlannedScene:
    layers: list[LayerSpec]
    safe_clock_rect: tuple[float, float, float, float]


def plan_layers(wallpaper_id: str, config: PipelineConfig) -> PlannedScene:
    layers = [
        LayerSpec(
            name="layer_0_far_bg",
            asset_path=f"{wallpaper_id}/layer_0_far_bg.webp",
            weight=config.depth_weights[0],
            blur_px=config.blur_px[0],
        ),
        LayerSpec(
            name="layer_1_deep_mid",
            asset_path=f"{wallpaper_id}/layer_1_deep_mid.webp",
            weight=config.depth_weights[1],
            blur_px=config.blur_px[1],
        ),
        LayerSpec(
            name="layer_2_near_mid",
            asset_path=f"{wallpaper_id}/layer_2_near_mid.webp",
            weight=config.depth_weights[2],
            blur_px=config.blur_px[2],
        ),
        LayerSpec(
            name="layer_3_hero_fg",
            asset_path=f"{wallpaper_id}/layer_3_hero_fg.webp",
            weight=config.depth_weights[3],
        ),
        LayerSpec(
            name="layer_4_front_fx",
            asset_path=f"{wallpaper_id}/layer_4_front_fx.webp",
            weight=config.depth_weights[4],
        ),
    ]
    return PlannedScene(layers=layers, safe_clock_rect=config.safe_clock_rect)
