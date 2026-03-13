from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

# ---------------------------------------------------------------------------
# ICC profile for color-managed output
# ---------------------------------------------------------------------------


def _build_srgb_icc_bytes() -> bytes:
    """Build an sRGB ICC profile for embedding in WebP output.

    This ensures viewers that support ICC profiles render colors correctly.
    Upgrade path: replace with a Display P3 .icc file for true wide-gamut.
    """
    profile = ImageCms.createProfile("sRGB")
    cms_profile = ImageCms.ImageCmsProfile(profile)
    return cms_profile.tobytes()


SRGB_ICC_PROFILE: bytes = _build_srgb_icc_bytes()

# ---------------------------------------------------------------------------
# sRGB linearization helpers (IEC 61966-2-1 exact EOTF)
# ---------------------------------------------------------------------------


def linearize(img: Image.Image) -> np.ndarray:
    """Convert an sRGB PIL Image to linear-light float32 RGBA array in [0,1]."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.concatenate([linear_rgb, arr[..., 3:4]], axis=-1)


def delinearize(arr: np.ndarray) -> Image.Image:
    """Convert a linear-light float32 RGBA array back to an sRGB PIL Image."""
    rgb = np.clip(arr[..., :3], 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)
    out = np.concatenate([srgb, np.clip(arr[..., 3:4], 0.0, 1.0)], axis=-1)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGBA")


def alpha_composite_linear(bg: Image.Image, fg: Image.Image) -> Image.Image:
    """Alpha-composite *fg* over *bg* in linear light (prevents dark halos)."""
    bg_lin = linearize(bg)
    fg_lin = linearize(fg)
    fg_a = fg_lin[..., 3:4]
    bg_a = bg_lin[..., 3:4]
    out_a = fg_a + bg_a * (1.0 - fg_a)
    safe_a = np.where(out_a > 0, out_a, 1.0)
    out_rgb = (fg_lin[..., :3] * fg_a + bg_lin[..., :3] * bg_a * (1.0 - fg_a)) / safe_a
    out = np.concatenate([out_rgb, out_a], axis=-1)
    return delinearize(out)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Input quality gate
# ---------------------------------------------------------------------------


def check_input_quality(
    path: Path,
    min_megapixels: float = 2.0,
    min_sharpness: float = 50.0,
) -> list[str]:
    """Return a list of warnings about source image quality.

    Checks:
    - Minimum resolution (megapixels).
    - Sharpness via Laplacian variance (detects soft/blurry sources).

    Returns an empty list if the image passes all gates.
    """
    import cv2 as _cv2

    warnings: list[str] = []
    with Image.open(path) as img:
        w, h = img.size
        mp = (w * h) / 1_000_000
        if mp < min_megapixels:
            warnings.append(
                f"Source resolution {w}×{h} ({mp:.1f} MP) is below "
                f"minimum {min_megapixels} MP — output may appear soft"
            )

    # Laplacian variance on a scaled-down greyscale copy (fast).
    bgr = _cv2.imread(str(path))
    if bgr is not None:
        grey = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
        # Down-sample for speed
        if grey.shape[0] > 512:
            scale = 512 / grey.shape[0]
            grey = _cv2.resize(grey, None, fx=scale, fy=scale)
        lap_var = float(_cv2.Laplacian(grey, _cv2.CV_64F).var())
        if lap_var < min_sharpness:
            warnings.append(
                f"Source sharpness ({lap_var:.1f}) is below "
                f"minimum {min_sharpness} — consider upscaling with Real-ESRGAN first"
            )

    return warnings


def load_image_canvas(path: Path, resolution: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.fit(
            source.convert("RGB"),
            resolution,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def mask_to_image(mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0.0, 1.0)
        mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def encode_webp(
    image: Image.Image,
    quality: int = 93,
    lossless: bool = False,
    icc_profile: bytes | None = None,
) -> bytes:
    buffer = BytesIO()
    kwargs: dict = {"format": "WEBP", "method": 6}
    if lossless:
        kwargs["lossless"] = True
    else:
        kwargs["quality"] = quality
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    image.save(buffer, **kwargs)
    return buffer.getvalue()
