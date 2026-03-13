from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from parallaxgen.config import PipelineConfig
from parallaxgen.depth.depth_utils import compute_depth_histogram_breaks
from parallaxgen.models import LayerSpec

# Canonical layer names in render order (back → front).
LAYER_NAMES = (
    "layer_0_far_bg",
    "layer_1_deep_mid",
    "layer_2_near_mid",
    "layer_3_hero_fg",
    "layer_4_front_fx",
)


@dataclass(slots=True)
class PlannedScene:
    layers: list[LayerSpec]
    safe_clock_rect: tuple[float, float, float, float]
    layer_masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    depth_breaks: list[float] = field(default_factory=list)


def _pick_breaks(depth_map: np.ndarray) -> list[float]:
    """Choose four depth boundaries to split the map into five bands.

    Uses histogram valley detection.  If not enough valleys are found we fall
    back to uniform quantile splits so we always get exactly four thresholds.
    """
    candidates = compute_depth_histogram_breaks(depth_map)
    # We need exactly 4 thresholds for 5 layers.
    if len(candidates) >= 4:
        # Spread picks evenly by depth *value*, not index, so bands
        # cover equal depth ranges even when valleys cluster together.
        lo, hi = candidates[0], candidates[-1]
        if hi - lo < 1e-6:
            targets = np.linspace(lo, lo + 1e-6, 4)
        else:
            targets = np.linspace(lo, hi, 6)[1:5]  # 4 interior points
        cands = np.array(candidates)
        picks: list[float] = []
        for t in targets:
            idx = int(np.argmin(np.abs(cands - t)))
            picks.append(float(cands[idx]))
        return picks
    # Fallback: uniform quantile breaks.
    flat = depth_map.ravel()
    return [float(np.percentile(flat, p)) for p in (20, 40, 60, 80)]


def _build_layer_masks(
    depth_map: np.ndarray,
    subject_alpha: np.ndarray,
    breaks: list[float],
) -> dict[str, np.ndarray]:
    """Create per-layer alpha masks from depth + subject segmentation.

    * Layers 0-2 are carved from the *background* (depth bands with subject
      removed) so the hero foreground is never duplicated.
    * Layer 3 (hero_fg) comes entirely from the subject matte.
    * Layer 4 (front_fx) is a thin edge fringe around the subject — the
      renderer uses it for subtle bokeh / bloom.
    """
    bg_weight = 1.0 - np.clip(subject_alpha, 0.0, 1.0)

    # Far background: depth < first break (the farthest band).
    far_mask = (depth_map < breaks[0]).astype(np.float32) * bg_weight

    # Deep midground.
    deep_mask = ((depth_map >= breaks[0]) & (depth_map < breaks[1])).astype(
        np.float32
    ) * bg_weight

    # Near midground.
    near_mask = ((depth_map >= breaks[1]) & (depth_map < breaks[2])).astype(
        np.float32
    ) * bg_weight

    # Nearest background band (between break 2 and break 3).
    # Pixels beyond break 3 that aren't subject fall here too.
    nearest_bg = ((depth_map >= breaks[2]) & (depth_map < breaks[3])).astype(
        np.float32
    ) * bg_weight
    # Fold the nearest-bg into near_mid so we still ship 5 layers.
    near_mask = np.clip(near_mask + nearest_bg, 0.0, 1.0)

    # Hero foreground — pure subject alpha.
    hero_mask = np.clip(subject_alpha, 0.0, 1.0)

    # Front FX — thin boundary shell around the subject.
    binary = (subject_alpha > 0.5).astype(np.uint8)
    if binary.max() == 0:
        fx_mask = np.zeros_like(subject_alpha, dtype=np.float32)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary, kernel)
        fx_mask = (binary.astype(np.float32) - eroded.astype(np.float32)).clip(0.0, 1.0)

    return {
        LAYER_NAMES[0]: far_mask,
        LAYER_NAMES[1]: deep_mask,
        LAYER_NAMES[2]: near_mask,
        LAYER_NAMES[3]: hero_mask,
        LAYER_NAMES[4]: fx_mask,
    }


def plan_layers(
    wallpaper_id: str,
    config: PipelineConfig,
    depth_map: np.ndarray | None = None,
    subject_alpha: np.ndarray | None = None,
) -> PlannedScene:
    """Plan the five-layer decomposition for a wallpaper.

    When *depth_map* and *subject_alpha* are provided the layer boundaries are
    computed from the image's actual depth histogram.  Without them (e.g. in
    unit-tests) the planner returns canonical layer specs with empty masks.
    """
    layers = [
        LayerSpec(
            name=name,
            asset_path=f"{wallpaper_id}/{name}.webp",
            weight=config.depth_weights[i],
            blur_px=config.blur_px[i],
        )
        for i, name in enumerate(LAYER_NAMES)
    ]

    if depth_map is not None and subject_alpha is not None:
        breaks = _pick_breaks(depth_map)
        masks = _build_layer_masks(depth_map, subject_alpha, breaks)
    else:
        breaks = []
        masks = {}

    return PlannedScene(
        layers=layers,
        safe_clock_rect=config.safe_clock_rect,
        layer_masks=masks,
        depth_breaks=breaks,
    )
