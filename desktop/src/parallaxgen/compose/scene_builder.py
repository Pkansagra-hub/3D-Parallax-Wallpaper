from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from parallaxgen.compose.layer_planner import SceneType, plan_layers
from parallaxgen.compose.occlusion_planner import (
    build_clock_occlusion_mask,
    compute_safe_clock_rect,
    derive_clock_layout,
)
from parallaxgen.compose.quality_scorer import score_scene
from parallaxgen.config import PipelineConfig
from parallaxgen.depth.depth_runner import DepthRunner
from parallaxgen.inpaint.inpainter import inpaint_background
from parallaxgen.models import (
    DEFAULT_CLOCK_WEIGHT,
    PACKAGE_CONTRACT,
    WallpaperMeta,
    WallpaperPackage,
)
from parallaxgen.preview.preview_renderer import render_preview_grid
from parallaxgen.segment.matte_refiner import refine_alpha
from parallaxgen.segment.subject_runner import SubjectRunner
from parallaxgen.utils.image_io import (
    SRGB_ICC_PROFILE,
    alpha_composite_linear,
    check_input_quality,
    encode_webp,
    load_image_canvas,
    mask_to_image,
)
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


def _apply_vignette(image: Image.Image, strength: float = 0.3) -> Image.Image:
    """Apply radial darkening towards corners for AMOLED depth effect."""
    w, h = image.size
    # Build radial gradient: 1.0 at centre, 0.0 at corners
    y, x = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    max_r = np.sqrt(cx**2 + cy**2)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r
    vignette = 1.0 - np.clip(r, 0.0, 1.0) * strength
    vignette_u8 = (vignette * 255).astype(np.uint8)
    mask = Image.fromarray(vignette_u8, mode="L")
    # Darken by compositing black through the inverse vignette mask
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    inv_mask = Image.fromarray(((1.0 - vignette) * 255).astype(np.uint8), mode="L")
    dark.putalpha(inv_mask)
    result = image.copy().convert("RGBA")
    result.alpha_composite(dark)
    return result


def _apply_chromatic_aberration(image: Image.Image, offset_px: int = 2) -> Image.Image:
    """Apply subtle RGB channel offset for cinematic lens feel."""
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    from PIL import ImageChops

    # Shift red channel left, blue channel right
    r_shifted = ImageChops.offset(r, -offset_px, 0)
    b_shifted = ImageChops.offset(b, offset_px, 0)
    return Image.merge("RGBA", (r_shifted, g, b_shifted, a))


def _alpha_layer(base_image: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
    rgba = base_image.convert("RGBA")
    rgba.putalpha(mask_to_image(alpha_mask))
    return rgba


def _render_assets(
    image_path: Path,
    config: PipelineConfig,
    depth_map: np.ndarray,
    subject_alpha: np.ndarray,
    layer_masks: dict[str, np.ndarray],
    safe_clock_rect: tuple[float, float, float, float],
    is_vista: bool = False,
) -> dict[str, bytes]:
    width, height = config.output_resolution
    base_image = load_image_canvas(image_path, config.output_resolution)

    # Always inpaint the base layer for clean parallax edge reveals.
    # PORTRAIT: remove the subject so shifting hero layer reveals clean bg.
    # VISTA: remove the hero depth band (nearest terrain) so far_bg reveals
    #        clean distant scenery when near layers shift during parallax —
    #        exactly like Apple's spatial depth wallpapers.
    if is_vista:
        inpaint_mask = layer_masks.get("layer_3_hero_fg", subject_alpha)
    else:
        inpaint_mask = subject_alpha

    if inpaint_mask.max() > 0.01:
        inpainted_bg = inpaint_background(
            base_image, inpaint_mask, depth_map=depth_map, method="auto"
        )
    else:
        inpainted_bg = base_image

    # --- Adaptive DOF blur --------------------------------------------------
    # Compute per-layer Gaussian blur from actual depth band centres.
    # Far layers get more blur (simulating camera defocus), hero is sharp.
    # In VISTA mode all 4 active layers are depth bands; in PORTRAIT mode
    # only layers 0-2 are background bands and layer 3 is the subject.
    _depth_centres = [0.85, 0.55, 0.30, 0.0, 0.0]  # far→front approx
    if depth_map is not None:
        _depth_band_keys = [
            ("layer_0_far_bg", 0),
            ("layer_1_deep_mid", 1),
            ("layer_2_near_mid", 2),
            ("layer_3_hero_fg", 3),
        ]
        for mask_key, idx in _depth_band_keys:
            m = layer_masks.get(mask_key)
            if m is not None and m.sum() > 0:
                _depth_centres[idx] = float(
                    np.average(depth_map, weights=np.maximum(m, 1e-8))
                )

    def _dof_blur(img: Image.Image, layer_idx: int) -> Image.Image:
        """Apply depth-adaptive Gaussian blur to a layer."""
        radius = _depth_centres[layer_idx] * config.max_blur_px
        if radius < 0.3:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Use layer masks from the planner (depth-driven decomposition).
    far_bg_mask = layer_masks.get(
        "layer_0_far_bg", np.ones((height, width), dtype=np.float32)
    )
    deep_mid_mask = layer_masks.get(
        "layer_1_deep_mid", np.zeros((height, width), dtype=np.float32)
    )
    near_mid_mask = layer_masks.get(
        "layer_2_near_mid", np.zeros((height, width), dtype=np.float32)
    )
    hero_fg_mask = layer_masks.get("layer_3_hero_fg", np.clip(subject_alpha, 0.0, 1.0))
    front_fx_mask = layer_masks.get(
        "layer_4_front_fx", np.zeros((height, width), dtype=np.float32)
    )

    # Clock occlusion: data-driven from actual subject matte.
    clock_occlusion_mask = build_clock_occlusion_mask(subject_alpha, safe_clock_rect)

    # Per-layer blurred images for adaptive DOF.
    far_bg_img = _apply_vignette(_dof_blur(inpainted_bg, 0)).convert("RGBA")
    deep_mid_img = _dof_blur(base_image, 1)
    near_mid_img = _dof_blur(base_image, 2)
    # VISTA: hero layer is a depth band that needs DOF blur too.
    # PORTRAIT: hero is the sharp subject — no blur.
    hero_img = _dof_blur(base_image, 3) if is_vista else base_image

    # Front FX with chromatic aberration for cinematic lens feel.
    front_fx_layer = _apply_chromatic_aberration(
        _alpha_layer(base_image, front_fx_mask)
    )

    # Build a preview composite in linear light (prevents dark alpha halos).
    preview = far_bg_img.copy()
    preview = alpha_composite_linear(preview, _alpha_layer(deep_mid_img, deep_mid_mask))
    preview = alpha_composite_linear(preview, _alpha_layer(near_mid_img, near_mid_mask))
    preview = alpha_composite_linear(preview, _alpha_layer(hero_img, hero_fg_mask))
    preview = alpha_composite_linear(preview, front_fx_layer)

    # Draw clock region indicator on preview.
    left = int(safe_clock_rect[0] * width)
    top = int(safe_clock_rect[1] * height)
    right = int(safe_clock_rect[2] * width)
    bottom = int(safe_clock_rect[3] * height)
    draw = ImageDraw.Draw(preview)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=18,
        outline=(255, 255, 255, 220),
        width=6,
    )
    draw.text((left + 24, top + 24), "12:40", fill=(255, 255, 255, 230))

    return {
        "layer_0_far_bg.webp": encode_webp(far_bg_img, icc_profile=SRGB_ICC_PROFILE),
        "layer_1_deep_mid.webp": encode_webp(
            _alpha_layer(deep_mid_img, deep_mid_mask), icc_profile=SRGB_ICC_PROFILE
        ),
        "layer_2_near_mid.webp": encode_webp(
            _alpha_layer(near_mid_img, near_mid_mask), icc_profile=SRGB_ICC_PROFILE
        ),
        "layer_3_hero_fg.webp": encode_webp(
            _alpha_layer(hero_img, hero_fg_mask),
            lossless=True,
            icc_profile=SRGB_ICC_PROFILE,
        ),
        "layer_4_front_fx.webp": encode_webp(
            front_fx_layer, lossless=True, icc_profile=SRGB_ICC_PROFILE
        ),
        "clock_occlusion_mask.webp": encode_webp(mask_to_image(clock_occlusion_mask)),
        "subject_mask.webp": encode_webp(mask_to_image(hero_fg_mask)),
        "depth_map.webp": encode_webp(mask_to_image(depth_map)),
        "preview.webp": encode_webp(preview, icc_profile=SRGB_ICC_PROFILE),
    }


def build_scene_package(
    image_path: Path,
    title: str | None = None,
    config: PipelineConfig | None = None,
    depth_runner: DepthRunner | None = None,
    subject_runner: SubjectRunner | None = None,
) -> WallpaperPackage:
    config = config or PipelineConfig()
    wallpaper_id = image_path.stem.lower().replace(" ", "_")
    t_total = time.perf_counter()

    # --- Input quality gate ---
    quality_warnings = check_input_quality(image_path)
    for qw in quality_warnings:
        logger.warning("%s: %s", wallpaper_id, qw)

    # --- Depth estimation ---
    t0 = time.perf_counter()
    if depth_runner is None:
        depth_runner = DepthRunner(
            model_name=config.depth_model,
            output_resolution=config.output_resolution,
        )
    depth_result = depth_runner.infer(image_path)
    logger.info("%s  depth  %.2fs", wallpaper_id, time.perf_counter() - t0)

    # --- Subject segmentation + matte refinement ---
    t0 = time.perf_counter()
    if subject_runner is None:
        subject_runner = SubjectRunner(
            model_name=config.segmentation_model,
        )
    subject_mask = subject_runner.infer(
        image_path, depth_result.width, depth_result.height
    )
    if subject_mask.is_landscape:
        refined_alpha = subject_mask.alpha  # already zeroed
        logger.info(
            "%s  landscape mode (no subject)  %.2fs",
            wallpaper_id,
            time.perf_counter() - t0,
        )
    else:
        refined_alpha = refine_alpha(subject_mask.alpha)
        logger.info("%s  segment  %.2fs", wallpaper_id, time.perf_counter() - t0)

    # --- Depth-driven layer planning ---
    t0 = time.perf_counter()
    planned_scene = plan_layers(
        wallpaper_id,
        config=config,
        depth_map=depth_result.depth_map,
        subject_alpha=refined_alpha,
        is_landscape=subject_mask.is_landscape,
    )
    is_vista = planned_scene.scene_type == SceneType.VISTA
    logger.info(
        "%s  plan  %.2fs  scene_type=%s",
        wallpaper_id,
        time.perf_counter() - t0,
        planned_scene.scene_type.value,
    )

    # --- Dynamic clock placement ---
    # For vista scenes, use the nearest depth band as the clock occluder
    # instead of the subject mask (which may be terrain, not a compact object).
    if is_vista:
        clock_occluder = planned_scene.layer_masks.get("layer_3_hero_fg", refined_alpha)
    else:
        clock_occluder = refined_alpha
    safe_rect = compute_safe_clock_rect(
        clock_occluder,
        config.safe_clock_rect,
        depth_map=depth_result.depth_map,
        scene_type=planned_scene.scene_type.value,
    )
    clock_anchor, clock_font_scale = derive_clock_layout(safe_rect)

    # --- Render all layer + support assets ---
    t0 = time.perf_counter()
    rendered_assets = _render_assets(
        image_path=image_path,
        config=config,
        depth_map=depth_result.depth_map,
        subject_alpha=refined_alpha,
        layer_masks=planned_scene.layer_masks,
        safe_clock_rect=safe_rect,
        is_vista=is_vista,
    )
    logger.info("%s  render  %.2fs", wallpaper_id, time.perf_counter() - t0)

    # --- Quality scoring ---
    quality_report = score_scene(
        depth_result.depth_map,
        refined_alpha,
        safe_rect,
        thresholds=config.quality_thresholds,
        layer_masks=planned_scene.layer_masks,
        scene_type=planned_scene.scene_type.value,
    )
    if quality_report.warnings:
        for w in quality_report.warnings:
            logger.warning("%s: %s", wallpaper_id, w)

    # --- QA preview grid (optional dev asset, not shipped to Android) ---
    base_image = load_image_canvas(image_path, config.output_resolution)
    qa_grid_bytes = render_preview_grid(
        base_image,
        depth_result.depth_map,
        refined_alpha,
        planned_scene.layer_masks,
        safe_rect,
    )

    meta = WallpaperMeta(
        wallpaper_id=wallpaper_id,
        target_device=config.target_device,
        resolution=(depth_result.width, depth_result.height),
        parallax_strength=config.parallax_strength,
        overscan=config.overscan,
        motion_profile=config.motion_profile,
        depth_weights=config.depth_weights.copy(),
        blur_px=config.blur_px.copy(),
        clock_weight=DEFAULT_CLOCK_WEIGHT,
        clock_font_scale=clock_font_scale,
        clock_anchor=clock_anchor,
        safe_clock_rect=safe_rect,
        has_clock_occlusion=True,
        subject_bbox=subject_mask.bbox,
        depth_model=config.depth_model,
        segmentation_model=config.segmentation_model,
        inpainted=not is_vista,
        quality=quality_report.to_dict(),
    )

    # Attach QA grid as an extra asset (written to disk but not in the
    # required contract — useful for visual inspection during development).
    rendered_assets["qa_grid.webp"] = qa_grid_bytes

    logger.info(
        "%s  TOTAL %.2fs  %s",
        wallpaper_id,
        time.perf_counter() - t_total,
        quality_report.summary_line(),
    )

    return WallpaperPackage(
        wallpaper_id=wallpaper_id,
        title=title or image_path.stem.replace("_", " ").title(),
        meta=meta,
        layers=planned_scene.layers,
        preview_asset=f"{wallpaper_id}/{PACKAGE_CONTRACT.required_support_assets[-1]}",
        rendered_assets=rendered_assets,
    )
