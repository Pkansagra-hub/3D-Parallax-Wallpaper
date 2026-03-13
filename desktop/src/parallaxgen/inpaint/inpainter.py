from __future__ import annotations

import contextlib
import logging
import os
import threading

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_hf_loading():
    """Suppress accelerate/transformers weight-materialisation tqdm spam."""
    prev = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        import transformers.utils.logging as hf_log

        hf_log.set_verbosity_error()
        yield
    finally:
        hf_log.set_verbosity_warning()
        if prev is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev


def _log_gpu_vram(label: str) -> None:
    """Log current GPU VRAM usage if CUDA is available."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            "GPU VRAM [%s]: %.2f GB allocated, %.2f GB reserved", label, alloc, reserved
        )


# ---------------------------------------------------------------------------
# Model singletons — loaded once, reused across the entire batch.
# Thread-safe via locks.
# ---------------------------------------------------------------------------

_flux_pipe = None
_flux_lock = threading.Lock()
_lama_model = None
_lama_lock = threading.Lock()

FLUX_INPAINT_REPO = "black-forest-labs/FLUX.1-Depth-dev"


def _get_flux_pipe():
    """Lazy-load FLUX.1 Depth inpainting pipeline (singleton, ~24 GB VRAM bf16)."""
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe

    with _flux_lock:
        if _flux_pipe is not None:
            return _flux_pipe

        from diffusers import FluxControlInpaintPipeline

        logger.info("Loading FLUX.1 Depth inpainting from %s …", FLUX_INPAINT_REPO)
        with _quiet_hf_loading():
            pipe = FluxControlInpaintPipeline.from_pretrained(
                FLUX_INPAINT_REPO,
                torch_dtype=torch.bfloat16,
            )
        pipe.to("cuda")
        _flux_pipe = pipe
        _log_gpu_vram("FLUX.1 loaded")
        return _flux_pipe


def _get_lama_model():
    """Lazy-load LaMa inpainting model (singleton, ~1.5 GB VRAM)."""
    global _lama_model
    if _lama_model is not None:
        return _lama_model

    with _lama_lock:
        if _lama_model is not None:
            return _lama_model

        from simple_lama_inpainting import SimpleLama  # type: ignore[import-untyped]

        logger.info("Loading LaMa inpainting model …")
        _lama_model = SimpleLama()
        logger.info("LaMa inpainting loaded")
        return _lama_model


# ---------------------------------------------------------------------------
# Mask preparation helpers
# ---------------------------------------------------------------------------


def _prepare_mask(
    subject_alpha: np.ndarray,
    erode_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inpaint masks from subject alpha.

    Dilation scales with image resolution so the inpaint region always
    extends far enough past the subject edge at any output size.

    Returns
    -------
    hard_mask : uint8 (0/255)
        Dilated binary mask for the inpaint region.
    soft_mask : float32 [0,1]
        Feathered version for blending the result back.
    """
    h, w = subject_alpha.shape
    binary = (subject_alpha > 0.25).astype(np.uint8)

    # Erode 3px — lets the inpainter see subject edge pixels as context.
    if erode_px > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        binary = cv2.erode(binary, k_erode)

    # Dilate — scaled to image resolution (~31px at 1440w, ~14px at 640w).
    dilate_px = max(12, min(h, w) // 100)
    k_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
    )
    dilated = cv2.dilate(binary, k_dilate)
    hard_mask = dilated * 255

    # Feather for blending — Gaussian blur creates a soft transition zone.
    feather_sigma = max(3.0, min(h, w) / 250.0)
    feather_ksize = int(feather_sigma * 6) | 1
    soft_mask = cv2.GaussianBlur(
        dilated.astype(np.float32),
        (feather_ksize, feather_ksize),
        feather_sigma,
    )
    return hard_mask, soft_mask


def _compute_context_crop(
    mask_u8: np.ndarray,
    img_h: int,
    img_w: int,
    padding_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    """Tight bounding box around the mask + padding, snapped to 8px grid.

    Returns (top, left, bottom, right) in pixel coordinates.
    """
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return (0, 0, img_h, img_w)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Add padding.
    bh, bw = y1 - y0, x1 - x0
    pad_y = int(bh * padding_ratio)
    pad_x = int(bw * padding_ratio)
    y0 = max(0, y0 - pad_y)
    y1 = min(img_h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(img_w, x1 + pad_x)

    # Make square-ish (SD models prefer square) — expand the shorter side.
    bh, bw = y1 - y0, x1 - x0
    if bh > bw:
        diff = bh - bw
        x0 = max(0, x0 - diff // 2)
        x1 = min(img_w, x1 + (diff - diff // 2))
    else:
        diff = bw - bh
        y0 = max(0, y0 - diff // 2)
        y1 = min(img_h, y1 + (diff - diff // 2))

    # Snap to 8px grid (required by SD).
    y0 = (y0 // 8) * 8
    x0 = (x0 // 8) * 8
    y1 = min(img_h, ((y1 + 7) // 8) * 8)
    x1 = min(img_w, ((x1 + 7) // 8) * 8)

    return (y0, x0, y1, x1)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _inpaint_cv2(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """OpenCV Telea — fast rough fill for the guidance pass."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, mask_u8, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def _inpaint_lama(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """LaMa ML inpainting — cached singleton, ~200ms on GPU.

    Mid-tier quality: far better than CV2 for structured backgrounds,
    fast enough for preview mode or CPU/low-VRAM fallback.
    """
    lama = _get_lama_model()
    img_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask_u8)
    result = lama(img_pil, mask_pil)
    return np.array(result.convert("RGB"))


# Scene-aware prompt for FLUX.1 — PORTRAIT context since VISTA skips inpainting.
# FLUX.1 uses T5-XXL which handles natural language well; no negative prompt needed.
_FLUX_PROMPT = (
    "seamless natural background continuation, photorealistic, "
    "consistent lighting and color temperature with surroundings, "
    "sharp details, high quality photograph, no people, no text, no watermark"
)


def _inpaint_flux(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    soft_mask: np.ndarray,
    depth_map: np.ndarray | None = None,
    guidance_rgb: np.ndarray | None = None,
    strength: float = 0.85,
    guidance_scale: float = 10.0,
    num_steps: int = 28,
) -> np.ndarray:
    """FLUX.1 Depth inpainting with depth control + context crop + soft paste-back.

    When *guidance_rgb* is provided (e.g. from a CV2 pre-fill), it's used as
    the init image so FLUX refines semantics while preserving color/tone.
    The *depth_map* (float32 [0,1]) is passed as a structural control signal
    so the inpainted region respects the scene's 3D geometry.
    The *soft_mask* is used for smooth blending at crop boundaries.
    """
    pipe = _get_flux_pipe()
    h, w = image_rgb.shape[:2]

    # Context crop — don't feed full wallpaper resolution to FLUX.
    crop = _compute_context_crop(mask_u8, h, w)
    y0, x0, y1, x1 = crop

    crop_mask = Image.fromarray(mask_u8[y0:y1, x0:x1])

    # Use CV2-prefilled guidance as init if available, else original.
    if guidance_rgb is not None:
        crop_guide = Image.fromarray(guidance_rgb[y0:y1, x0:x1])
    else:
        crop_guide = Image.fromarray(image_rgb[y0:y1, x0:x1])

    # Build depth control image from our precomputed depth map.
    if depth_map is not None:
        depth_crop = depth_map[y0:y1, x0:x1]
        depth_u8 = (np.clip(depth_crop, 0.0, 1.0) * 255).astype(np.uint8)
        control_image = Image.fromarray(depth_u8, mode="L").convert("RGB")
    else:
        # Fallback: flat mid-gray depth (neutral structural guidance).
        cw_raw, ch_raw = crop_guide.size
        control_image = Image.new("RGB", (cw_raw, ch_raw), (128, 128, 128))

    # Resize to max 1024 for quality/speed balance.
    cw, ch = crop_guide.size
    scale = min(1024 / max(cw, ch), 1.0)
    gen_w = ((int(cw * scale) + 7) // 8) * 8
    gen_h = ((int(ch * scale) + 7) // 8) * 8
    gen_w = max(gen_w, 64)
    gen_h = max(gen_h, 64)

    crop_guide_resized = crop_guide.resize((gen_w, gen_h), Image.LANCZOS)
    crop_mask_resized = crop_mask.resize((gen_w, gen_h), Image.NEAREST)
    control_resized = control_image.resize((gen_w, gen_h), Image.LANCZOS)

    with torch.inference_mode():
        result = pipe(
            prompt=_FLUX_PROMPT,
            image=crop_guide_resized,
            control_image=control_resized,
            mask_image=crop_mask_resized,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="pil",
        ).images[0]

    # Resize back to crop size.
    result_full = result.resize((cw, ch), Image.LANCZOS)
    result_np = np.array(result_full)

    # Soft blend crop result back — avoids hard rectangular seam at crop edge.
    output = image_rgb.copy()
    crop_soft = soft_mask[y0:y1, x0:x1, None]
    output[y0:y1, x0:x1] = np.clip(
        image_rgb[y0:y1, x0:x1].astype(np.float32) * (1.0 - crop_soft)
        + result_np.astype(np.float32) * crop_soft,
        0,
        255,
    ).astype(np.uint8)
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inpaint_background(
    background_image: Image.Image,
    subject_alpha: np.ndarray,
    depth_map: np.ndarray | None = None,
    method: str = "auto",
) -> Image.Image:
    """Fill the hole left by the subject — production tiered pipeline.

    Tier resolution (method="auto"):
        FLUX.1 Depth + CUDA → two-pass (CV2 guidance → FLUX depth-aware)
        LaMa installed      → single-pass LaMa (fast, good for textures)
        Else                → CV2 Telea fallback

    Pipeline (FLUX path):
        1. Erode mask 3px → resolution-scaled dilate → feather
        2. Pass 1: CV2 Telea rough fill (color/tone guidance, <50ms)
        3. Pass 2: FLUX.1 Depth inpaint on context-cropped region with
           CV2 as init, depth map as control (strength=0.85)
        4. Soft blend at crop paste-back (no rectangular seam)
        5. Secondary feathered blend on full image (safety pass)

    Parameters
    ----------
    depth_map : float32 [0,1] or None
        Precomputed depth map (0=near, 1=far). Passed to FLUX.1 Depth
        as structural control so inpainted regions respect scene geometry.

    Explicit methods: "flux", "lama", "cv2".
    """
    bg_rgb = np.array(background_image.convert("RGB"))
    hard_mask, soft_mask = _prepare_mask(subject_alpha)

    if method == "auto":
        if torch.cuda.is_available():
            try:
                import diffusers  # noqa: F401

                method = "flux"
            except ImportError:
                logger.warning("diffusers not installed — FLUX unavailable, using LaMa")
                method = "lama"
        else:
            logger.info("No CUDA — using LaMa inpainting")
            method = "lama"

    logger.info("Inpainting method: %s", method)

    if method in ("flux", "sdxl"):  # "sdxl" kept for backward compat
        try:
            # Pass 1: rough CV2 fill for guidance.
            rough_fill = _inpaint_cv2(bg_rgb, hard_mask)

            # Pass 2: FLUX.1 Depth refines semantics using the rough fill
            # as init and our depth map as structural control.
            refined = _inpaint_flux(
                bg_rgb,
                hard_mask,
                soft_mask=soft_mask,
                depth_map=depth_map,
                guidance_rgb=rough_fill,
                strength=0.85,
                guidance_scale=10.0,
                num_steps=28,
            )

            # Secondary feathered blend on full image — safety pass
            # in case crop boundary didn't fully cover the transition zone.
            alpha_3ch = soft_mask[:, :, None]
            blended = (
                bg_rgb.astype(np.float32) * (1.0 - alpha_3ch)
                + refined.astype(np.float32) * alpha_3ch
            )
            return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

        except Exception as e:
            logger.error(
                "FLUX inpainting FAILED: %s — falling back to LaMa. "
                "This means your output is NOT using SOTA inpainting!",
                e,
            )
            method = "lama"

    if method == "lama":
        try:
            result = _inpaint_lama(bg_rgb, hard_mask)
            # Feathered blend for clean transition.
            alpha_3ch = soft_mask[:, :, None]
            blended = (
                bg_rgb.astype(np.float32) * (1.0 - alpha_3ch)
                + result.astype(np.float32) * alpha_3ch
            )
            return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        except Exception as e:
            logger.warning("LaMa inpainting failed (%s), falling back to cv2", e)
            method = "cv2"

    if method == "cv2":
        result = _inpaint_cv2(bg_rgb, hard_mask)
        return Image.fromarray(result)

    logger.warning("Unknown inpaint method '%s', returning original", method)
    return background_image
