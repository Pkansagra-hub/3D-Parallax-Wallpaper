from __future__ import annotations

# Avoid circular import at module level — only needed inside render_preview_summary.
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from parallaxgen.utils.image_io import encode_webp

if TYPE_CHECKING:

    pass

# Layer colour key for the assignment visualisation (RGBA).
_LAYER_COLOURS = [
    (50, 80, 180, 140),  # far_bg — blue
    (50, 180, 120, 140),  # deep_mid — teal
    (200, 200, 50, 140),  # near_mid — yellow
    (220, 80, 40, 180),  # hero_fg — red
    (220, 40, 200, 160),  # front_fx — magenta
]


def render_preview_grid(
    base_image: Image.Image,
    depth_map: np.ndarray,
    subject_alpha: np.ndarray,
    layer_masks: dict[str, np.ndarray],
    safe_clock_rect: tuple[float, float, float, float],
) -> bytes:
    """Render a 2×2 QA preview grid and return it as WebP bytes.

    Quadrants:
      TL — original image with clock safe-rect overlay
      TR — depth map as a plasma colour-map
      BL — subject alpha matte
      BR — colour-coded layer assignment
    """
    w, h = base_image.size
    thumb = (w // 2, h // 2)

    # --- TL: original + clock overlay ---
    tl = base_image.copy().resize(thumb, Image.Resampling.LANCZOS).convert("RGBA")
    draw = ImageDraw.Draw(tl)
    lx, ty, rx, by = (
        int(safe_clock_rect[0] * thumb[0]),
        int(safe_clock_rect[1] * thumb[1]),
        int(safe_clock_rect[2] * thumb[0]),
        int(safe_clock_rect[3] * thumb[1]),
    )
    draw.rounded_rectangle(
        (lx, ty, rx, by), radius=10, outline=(255, 255, 255, 200), width=4
    )
    draw.text((lx + 12, ty + 8), "12:40", fill=(255, 255, 255, 220))

    # --- TR: depth heatmap (plasma-like: blue→green→yellow→red) ---
    depth_u8 = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
    depth_rgb = _pseudo_plasma(depth_u8)
    tr = (
        Image.fromarray(depth_rgb)
        .resize(thumb, Image.Resampling.LANCZOS)
        .convert("RGBA")
    )

    # --- BL: subject alpha ---
    alpha_u8 = (np.clip(subject_alpha, 0, 1) * 255).astype(np.uint8)
    bl = Image.fromarray(alpha_u8, mode="L").resize(thumb, Image.Resampling.LANCZOS)
    bl = bl.convert("RGBA")

    # --- BR: colour-coded layer assignment ---
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    ordered_names = [
        "layer_0_far_bg",
        "layer_1_deep_mid",
        "layer_2_near_mid",
        "layer_3_hero_fg",
        "layer_4_front_fx",
    ]
    for i, name in enumerate(ordered_names):
        mask = layer_masks.get(name)
        if mask is None:
            continue
        colour = np.array(_LAYER_COLOURS[i], dtype=np.uint8)
        overlay = (mask > 0.15).astype(np.uint8)
        for c in range(4):
            canvas[:, :, c] = np.where(overlay, colour[c], canvas[:, :, c])
    br = Image.fromarray(canvas, "RGBA").resize(thumb, Image.Resampling.LANCZOS)

    # --- Stitch 2×2 ---
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    grid.paste(tl, (0, 0))
    grid.paste(tr, (thumb[0], 0))
    grid.paste(bl, (0, thumb[1]))
    grid.paste(br, (thumb[0], thumb[1]))

    return encode_webp(grid.convert("RGB"), quality=85)


def _pseudo_plasma(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel uint8 image to a 3-channel plasma-like map."""
    h, w = gray.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    t = gray.astype(np.float32) / 255.0
    # R: ramp up in the upper half
    out[:, :, 0] = np.clip(t * 2.0 - 0.5, 0, 1) * 255
    # G: peak in the middle
    out[:, :, 1] = np.clip(1.0 - np.abs(t - 0.5) * 3.0, 0, 1) * 255
    # B: ramp down
    out[:, :, 2] = np.clip(1.0 - t * 1.8, 0, 1) * 255
    return out


def render_preview_summary(
    image_path: "Path",
    include_clock: bool = False,
    config: "PipelineConfig | None" = None,
) -> dict[str, object]:
    """Return a JSON-serialisable dict describing the pipeline preview for *image_path*.

    This is a lightweight introspection tool — it runs depth + segmentation
    but does NOT render the full package.  Useful for quick CLI inspection.
    """
    from pathlib import Path as _Path

    from parallaxgen.compose.occlusion_planner import compute_safe_clock_rect
    from parallaxgen.compose.quality_scorer import score_scene
    from parallaxgen.config import PipelineConfig as _Cfg
    from parallaxgen.depth.depth_runner import DepthRunner
    from parallaxgen.segment.matte_refiner import refine_alpha
    from parallaxgen.segment.subject_runner import SubjectRunner

    config = config or _Cfg()
    image_path = _Path(image_path)

    depth = DepthRunner(
        model_name=config.depth_model,
        output_resolution=config.output_resolution,
    ).infer(image_path)

    subject = SubjectRunner(model_name=config.segmentation_model).infer(
        image_path, depth.width, depth.height
    )
    alpha = refine_alpha(subject.alpha)
    safe_rect = compute_safe_clock_rect(alpha, config.safe_clock_rect)

    quality = score_scene(depth.depth_map, alpha, safe_rect, config.quality_thresholds)

    summary: dict[str, object] = {
        "image": image_path.name,
        "resolution": [depth.width, depth.height],
        "depth_model": config.depth_model,
        "segmentation_model": config.segmentation_model,
        "subject_bbox": list(subject.bbox),
        "quality": quality.to_dict(),
    }
    if include_clock:
        summary["safe_clock_rect"] = list(safe_rect)
    return summary
