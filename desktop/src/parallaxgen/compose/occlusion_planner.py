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


def _clip_rect(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[float, float, float, float]:
    left = float(np.clip(left, 0.0, 1.0))
    top = float(np.clip(top, 0.0, 1.0))
    right = float(np.clip(right, 0.0, 1.0))
    bottom = float(np.clip(bottom, 0.0, 1.0))
    return (left, top, max(left + 1e-4, right), max(top + 1e-4, bottom))


def _rect_region(
    image: np.ndarray,
    rect: tuple[float, float, float, float],
) -> np.ndarray:
    h, w = image.shape[:2]
    l, t, r, b = rect
    li, ti = int(l * w), int(t * h)
    ri, bi = int(r * w), int(b * h)
    if ri <= li or bi <= ti:
        return np.zeros((0, 0), dtype=image.dtype)
    return image[ti:bi, li:ri]


def derive_clock_layout(
    safe_rect: tuple[float, float, float, float],
) -> tuple[tuple[float, float], float]:
    """Derive clock anchor and font scale from the chosen safe rect.

    The clock occupies a compact bounding window in metadata, while the actual
    digits sit slightly below the rect centre.  Font scale is tied to the rect
    width so Android can resize the clock proportionally across scenes.
    """
    l, t, r, b = safe_rect
    rect_w = max(r - l, 1e-4)
    rect_h = max(b - t, 1e-4)
    anchor = (float((l + r) * 0.5), float(t + rect_h * 0.56))
    font_scale = float(np.clip(rect_w * 0.92, 0.28, 0.56))
    return anchor, font_scale


def compute_safe_clock_rect(
    subject_alpha: np.ndarray,
    preferred_rect: tuple[float, float, float, float],
    threshold: float = 0.3,
    depth_map: np.ndarray | None = None,
    scene_type: str = "portrait",
) -> tuple[float, float, float, float]:
    """Find a compact, readable rectangle for clock placement.

    Rather than scanning full-width horizontal bands, evaluate many candidate
    rectangles across the upper/mid frame and score them by:
    - subject occlusion balance (some overlap is acceptable and often desirable)
    - depth smoothness/readability (prefer uniform background)
    - mean depth for vista scenes (prefer farther, more sky/open-space zones)
    - centrality / top bias for lock-screen ergonomics

    This better matches the actual render model where the clock is its own
    intermediate plane, not a full-width strip.
    """
    h, w = subject_alpha.shape

    def _occlusion_score(rect: tuple[float, float, float, float]) -> float:
        region = _rect_region(subject_alpha, rect)
        if region.size == 0:
            return 1.0
        return 1.0 - float((region > threshold).mean())

    def _depth_stats(rect: tuple[float, float, float, float]) -> tuple[float, float]:
        if depth_map is None:
            return (0.5, 0.0)
        region = _rect_region(depth_map, rect)
        if region.size == 0:
            return (0.5, 0.0)
        return (float(region.mean()), float(region.std()))

    pref_l, pref_t, pref_r, pref_b = preferred_rect
    pref_cx = (pref_l + pref_r) * 0.5

    width_candidates = [0.34, 0.40, 0.46]
    height_candidates = [0.10, 0.115, 0.13]
    x_centres = [pref_cx, 0.50, 0.42, 0.58]
    if scene_type == "vista":
        y_centres = np.linspace(0.16, 0.42, 8)
        target_y = 0.24
    else:
        y_centres = np.linspace(0.14, 0.34, 7)
        target_y = 0.22

    best_rect = preferred_rect
    best_score = -1e9

    for rect_w in width_candidates:
        for rect_h in height_candidates:
            for cx in x_centres:
                for cy in y_centres:
                    rect = _clip_rect(
                        cx - rect_w * 0.5,
                        cy - rect_h * 0.5,
                        cx + rect_w * 0.5,
                        cy + rect_h * 0.5,
                    )
                    occ_clear = _occlusion_score(rect)
                    overlap_ratio = 1.0 - occ_clear
                    mean_depth, depth_std = _depth_stats(rect)
                    top_bias = 1.0 - min(abs(cy - target_y) / 0.24, 1.0)
                    centre_bias = 1.0 - min(abs(cx - 0.50) / 0.20, 1.0)
                    smooth_bg = 1.0 - min(depth_std / 0.20, 1.0)

                    target_overlap = 0.20 if scene_type == "vista" else 0.14
                    overlap_balance = 1.0 - min(
                        abs(overlap_ratio - target_overlap) / 0.22,
                        1.0,
                    )
                    visible_ratio = occ_clear
                    penalty = -3.0 if visible_ratio < 0.18 else 0.0
                    vista_depth_bonus = mean_depth if scene_type == "vista" else 0.0
                    score = (
                        overlap_balance * 1.8
                        + visible_ratio * 2.0
                        + smooth_bg * 1.2
                        + vista_depth_bonus * 0.8
                        + top_bias * 0.8
                        + centre_bias * 0.3
                        + penalty
                    )

                    if score > best_score:
                        best_score = score
                        best_rect = rect

    return best_rect
