from __future__ import annotations

import cv2
import numpy as np


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Normalise a depth map to ``[0.0, 1.0]``."""
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth_map, dtype=np.float32)
    return ((depth_map - d_min) / (d_max - d_min)).astype(np.float32)


def smooth_depth(depth_map: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smooth to reduce high-frequency noise while keeping large structures."""
    ksize = int(sigma * 6) | 1  # must be odd
    return cv2.GaussianBlur(depth_map.astype(np.float32), (ksize, ksize), sigma).astype(
        np.float32
    )


def edge_preserving_smooth(
    depth_map: np.ndarray,
    sigma_space: float = 30.0,
    sigma_color: float = 0.15,
) -> np.ndarray:
    """Bilateral filter: smooth flat regions, preserve depth discontinuities."""
    depth_u8 = np.clip(depth_map * 255, 0, 255).astype(np.uint8)
    smoothed = cv2.bilateralFilter(
        depth_u8,
        d=-1,
        sigmaColor=sigma_color * 255,
        sigmaSpace=sigma_space,
    )
    return smoothed.astype(np.float32) / 255.0


def compute_depth_histogram_breaks(
    depth_map: np.ndarray,
    n_bins: int = 256,
    kernel_size: int = 7,
) -> list[float]:
    """Find natural depth-band boundaries via histogram valley detection.

    Returns a sorted list of depth values (in ``[0, 1]``) corresponding to
    local minima in the smoothed depth histogram.  These are good candidates
    for splitting the scene into layers without cutting through a dominant
    depth cluster.
    """
    flat = depth_map.ravel()
    # Ignore exact-zero pixels (masked-out regions)
    flat = flat[flat > 0.01]
    if flat.size == 0:
        return []

    hist, bin_edges = np.histogram(flat, bins=n_bins, range=(0.0, 1.0))
    kernel = np.ones(kernel_size) / kernel_size
    smooth_hist = np.convolve(hist.astype(np.float64), kernel, mode="same")

    valleys: list[float] = []
    for i in range(1, len(smooth_hist) - 1):
        if smooth_hist[i] < smooth_hist[i - 1] and smooth_hist[i] < smooth_hist[i + 1]:
            valleys.append(float(bin_edges[i]))

    return sorted(valleys)
