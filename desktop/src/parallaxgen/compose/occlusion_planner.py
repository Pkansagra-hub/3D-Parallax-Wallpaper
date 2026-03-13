from __future__ import annotations

import cv2
import numpy as np


def build_clock_occlusion_mask(
    subject_alpha: np.ndarray,
    clock_rect: tuple[float, float, float, float],
    dilation_px: int = 12,
    threshold: float = 0.3,
) -> np.ndarray:
    """Build a per-pixel mask of where the subject occludes the clock.

    The Android renderer uses this mask to cut holes in the clock texture so
    the hero foreground appears to pass **in front of** the clock digits —
    exactly like the Apple spatial lock-screen effect.

    The mask covers the full image but only the region inside *clock_rect* is
    meaningful.  White (255) = subject occludes the clock here.

    Parameters
    ----------
    subject_alpha:
        Refined subject matte, float32 [0, 1], shape (H, W).
    clock_rect:
        Normalised (left, top, right, bottom) of the clock rendering area.
    dilation_px:
        Extra safety margin so the occlusion mask extends slightly past the
        raw subject boundary, preventing thin halo artifacts.
    threshold:
        Subject alpha value above which a pixel is considered foreground.
    """
    h, w = subject_alpha.shape
    l, t, r, b = clock_rect
    li, ti, ri, bi = int(l * w), int(t * h), int(r * w), int(b * h)

    # Subject pixels inside the clock bounding box
    mask = np.zeros((h, w), dtype=np.uint8)
    region = subject_alpha[ti:bi, li:ri]
    mask[ti:bi, li:ri] = (region > threshold).astype(np.uint8) * 255

    # Dilate for safety margin
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def compute_safe_clock_rect(
    subject_alpha: np.ndarray,
    preferred_rect: tuple[float, float, float, float],
    threshold: float = 0.3,
) -> tuple[float, float, float, float]:
    """Find the best unobstructed rectangle for clock placement.

    If the preferred rect is mostly clear (<15 % occluded), return it as-is.
    Otherwise scan the top 40 % of the image for the tallest contiguous
    horizontal band with low occlusion and return that.
    """
    h, w = subject_alpha.shape
    l, t, r, b = preferred_rect
    region = subject_alpha[int(t * h) : int(b * h), int(l * w) : int(r * w)]
    occlusion_ratio = float((region > threshold).mean())

    if occlusion_ratio < 0.15:
        return preferred_rect

    # Scan the top 40 % of the frame for a clear band
    scan_bottom = int(h * 0.40)
    top_region = subject_alpha[:scan_bottom, :]
    row_occlusion = (top_region > threshold).mean(axis=1)

    clear = row_occlusion < 0.10
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, is_clear in enumerate(clear):
        if is_clear:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    min_band_height = int(h * 0.06)
    if best_len >= min_band_height:
        return (0.08, best_start / h, 0.92, (best_start + best_len) / h)

    # --- Third fallback: scan bottom 20 % of the frame ---
    bot_start = int(h * 0.80)
    bot_region = subject_alpha[bot_start:, :]
    bot_row_occ = (bot_region > threshold).mean(axis=1)
    bot_clear = bot_row_occ < 0.10
    b_best_start, b_best_len, b_cur_start, b_cur_len = 0, 0, 0, 0
    for i, is_clear in enumerate(bot_clear):
        if is_clear:
            if b_cur_len == 0:
                b_cur_start = i
            b_cur_len += 1
            if b_cur_len > b_best_len:
                b_best_start, b_best_len = b_cur_start, b_cur_len
        else:
            b_cur_len = 0

    if b_best_len >= min_band_height:
        abs_start = bot_start + b_best_start
        return (0.08, abs_start / h, 0.92, (abs_start + b_best_len) / h)

    # No unobstructed zone found anywhere.  Return preferred rect but log
    # a warning — the quality scorer will flag clock_readability.
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "No clear clock zone found (top + bottom scan failed); "
        "returning preferred rect as fallback"
    )
    return preferred_rect
