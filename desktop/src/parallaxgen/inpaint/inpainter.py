from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singletons — loaded once, reused across the entire batch.
# Thread-safe via locks.
# ---------------------------------------------------------------------------

_sdxl_pipe = None
_sdxl_lock = threading.Lock()
_lama_model = None
_lama_lock = threading.Lock()

SDXL_INPAINT_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def _get_sdxl_pipe():
    """Lazy-load SDXL inpainting pipeline (singleton, ~6 GB VRAM)."""
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    with _sdxl_lock:
        if _sdxl_pipe is not None:
            return _sdxl_pipe

        from diffusers import StableDiffusionXLInpaintPipeline

        logger.info("Loading SDXL inpainting from %s …", SDXL_INPAINT_REPO)
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            SDXL_INPAINT_REPO,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.to("cuda")
        # Enable memory-efficient attention if available.
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        _sdxl_pipe = pipe
        logger.info("SDXL inpainting loaded")
        return _sdxl_pipe


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


# Scene-aware prompts for SDXL — derived from PORTRAIT context since
# VISTA scenes skip inpainting entirely.
_SDXL_PROMPT = (
    "clean natural background, seamless continuation of surroundings, "
    "high quality wallpaper photograph, no people, no text, no objects, "
    "consistent lighting, sharp details"
)
_SDXL_NEGATIVE = (
    "text, watermark, logo, person, face, hand, body, fingers, limbs, "
    "artifact, blur, noise, low quality, jpeg artifacts, deformed, "
    "extra objects, unnatural colors, seam, border"
)


def _inpaint_sdxl(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    soft_mask: np.ndarray,
    guidance_rgb: np.ndarray | None = None,
    strength: float = 0.58,
    guidance_scale: float = 8.0,
    num_steps: int = 30,
) -> np.ndarray:
    """SDXL inpainting with context crop + guidance image + soft paste-back.

    When *guidance_rgb* is provided (e.g. from a CV2 pre-fill), it's used as
    the init image so SD only corrects semantics rather than generating from
    noise.  The *soft_mask* is used for blending the crop back into the full
    image to avoid hard rectangular seams at crop boundaries.
    """
    pipe = _get_sdxl_pipe()
    h, w = image_rgb.shape[:2]

    # Context crop — don't feed 1440×3120 to SD.
    crop = _compute_context_crop(mask_u8, h, w)
    y0, x0, y1, x1 = crop

    crop_img = Image.fromarray(image_rgb[y0:y1, x0:x1])
    crop_mask = Image.fromarray(mask_u8[y0:y1, x0:x1])

    # If we have a guidance (pre-filled) image, use it as init.
    if guidance_rgb is not None:
        crop_guide = Image.fromarray(guidance_rgb[y0:y1, x0:x1])
    else:
        crop_guide = crop_img

    # SD needs specific sizes — resize to max 1024 for quality/speed balance.
    cw, ch = crop_img.size
    scale = min(1024 / max(cw, ch), 1.0)
    sd_w = ((int(cw * scale) + 7) // 8) * 8
    sd_h = ((int(ch * scale) + 7) // 8) * 8
    sd_w = max(sd_w, 64)
    sd_h = max(sd_h, 64)

    crop_img_resized = crop_img.resize((sd_w, sd_h), Image.LANCZOS)
    crop_mask_resized = crop_mask.resize((sd_w, sd_h), Image.NEAREST)
    crop_guide_resized = crop_guide.resize((sd_w, sd_h), Image.LANCZOS)

    with torch.inference_mode():
        result = pipe(
            prompt=_SDXL_PROMPT,
            negative_prompt=_SDXL_NEGATIVE,
            image=crop_guide_resized,
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
        0, 255,
    ).astype(np.uint8)
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inpaint_background(
    background_image: Image.Image,
    subject_alpha: np.ndarray,
    method: str = "auto",
) -> Image.Image:
    """Fill the hole left by the subject — production tiered pipeline.

    Tier resolution (method="auto"):
        SDXL + CUDA available → two-pass (CV2 guidance → SDXL semantic fix)
        LaMa installed        → single-pass LaMa (fast, good for textures)
        Else                  → CV2 Telea fallback

    Pipeline (SDXL path):
        1. Erode mask 3px → resolution-scaled dilate → feather
        2. Pass 1: CV2 Telea rough fill (color/tone guidance, <50ms)
        3. Pass 2: SDXL inpaint on context-cropped region with CV2 as
           init image (strength=0.58 → SD fixes semantics, keeps tone)
        4. Soft blend at crop paste-back (no rectangular seam)
        5. Secondary feathered blend on full image (safety pass)

    Explicit methods: "sdxl", "lama", "cv2".
    """
    bg_rgb = np.array(background_image.convert("RGB"))
    hard_mask, soft_mask = _prepare_mask(subject_alpha)

    if method == "auto":
        if torch.cuda.is_available():
            try:
                import diffusers  # noqa: F401
                method = "sdxl"
            except ImportError:
                method = "lama"
        else:
            method = "lama"

    if method == "sdxl":
        try:
            # Pass 1: rough CV2 fill for guidance.
            rough_fill = _inpaint_cv2(bg_rgb, hard_mask)

            # Pass 2: SDXL refines semantics using the rough fill as init.
            # soft_mask is passed in so paste-back uses soft blending.
            refined = _inpaint_sdxl(
                bg_rgb,
                hard_mask,
                soft_mask=soft_mask,
                guidance_rgb=rough_fill,
                strength=0.58,
                guidance_scale=8.0,
                num_steps=30,
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
            logger.warning("SDXL inpainting failed (%s), falling back to LaMa", e)
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
