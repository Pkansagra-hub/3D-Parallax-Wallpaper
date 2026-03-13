from __future__ import annotations

from dataclasses import dataclass

from parallaxgen.models import LayerSpec


@dataclass(slots=True)
class PlannedScene:
    layers: list[LayerSpec]
    safe_clock_rect: tuple[float, float, float, float]


def plan_layers(wallpaper_id: str) -> PlannedScene:
    layers = [
        LayerSpec(
            name="layer_0_far_bg",
            asset_path=f"{wallpaper_id}/layer_0_far_bg.webp",
            weight=0.08,
            blur_px=1.2,
        ),
        LayerSpec(
            name="layer_1_deep_mid",
            asset_path=f"{wallpaper_id}/layer_1_deep_mid.webp",
            weight=0.18,
            blur_px=0.8,
        ),
        LayerSpec(
            name="layer_2_near_mid",
            asset_path=f"{wallpaper_id}/layer_2_near_mid.webp",
            weight=0.32,
            blur_px=0.3,
        ),
        LayerSpec(
            name="layer_3_hero_fg",
            asset_path=f"{wallpaper_id}/layer_3_hero_fg.webp",
            weight=0.48,
        ),
        LayerSpec(
            name="layer_4_front_fx",
            asset_path=f"{wallpaper_id}/layer_4_front_fx.webp",
            weight=0.62,
        ),
    ]
    return PlannedScene(layers=layers, safe_clock_rect=(0.16, 0.07, 0.84, 0.30))
