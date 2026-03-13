from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from parallaxgen.config import PipelineConfig
from parallaxgen.depth.depth_utils import edge_preserving_smooth
from parallaxgen.models import LayerSpec

# Canonical layer names in render order (back → front).
LAYER_NAMES = (
    "layer_0_far_bg",
    "layer_1_deep_mid",
    "layer_2_near_mid",
    "layer_3_hero_fg",
    "layer_4_front_fx",
)


class SceneType(str, Enum):
    """Decomposition strategy chosen by the scene classifier."""

    PORTRAIT = "portrait"  # Distinct compact subject → extract, inpaint, depth-split bg
    VISTA = "vista"  # Landscape / terrain → pure depth decomposition, no inpaint


@dataclass(slots=True)
class PlannedScene:
    layers: list[LayerSpec]
    safe_clock_rect: tuple[float, float, float, float]
    scene_type: SceneType = SceneType.PORTRAIT
    layer_masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    depth_breaks: list[float] = field(default_factory=list)


def _gaussian_mask(depth_map: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-3)
    return np.exp(-0.5 * ((depth_map - centre) / sigma) ** 2).astype(np.float32)


def _largest_components_mask(binary: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max(initial=0))
    keep = np.zeros_like(binary)
    for idx, area in enumerate(component_areas, start=1):
        if area >= max(min_area, int(largest_area * 0.18)):
            keep[labels == idx] = 1
    return keep


def _clean_subject_alpha(subject_alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(subject_alpha, 0.0, 1.0).astype(np.float32)
    binary = (alpha > 0.18).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(alpha)

    h, w = alpha.shape
    kernel_size = max(3, ((min(h, w) // 180) * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = _largest_components_mask(binary, min_area=max(64, int(h * w * 0.0012)))

    feather = cv2.GaussianBlur(
        binary.astype(np.float32),
        (0, 0),
        sigmaX=max(1.5, min(h, w) / 320.0),
        sigmaY=max(1.5, min(h, w) / 320.0),
    )
    return np.clip(alpha * feather, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Focus-driven depth decomposition via K-means
# ---------------------------------------------------------------------------


def _find_depth_centres(flat: np.ndarray, n: int = 4) -> list[float]:
    """Find *n* optimal depth band centres via K-means clustering.

    K-means naturally adapts to the depth distribution: bimodal scenes (sky +
    ground) get centres that respect both modes; continuous gradients get
    evenly spaced centres.  No histogram valley detection, no bimodal
    heuristics — just let the data speak.

    Returns centres sorted descending (far → near).
    """
    if flat.size < n:
        return list(np.linspace(1.0, 0.0, n))

    # Subsample for speed (cv2.kmeans is O(n·K·iters)).
    rng = np.random.default_rng(42)
    sample = rng.choice(flat, min(20000, flat.size), replace=False)
    sample_f32 = sample.reshape(-1, 1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, _, raw_centres = cv2.kmeans(
        sample_f32, n, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    centres = sorted(raw_centres.flatten().tolist(), reverse=True)

    # Enforce minimum separation so adjacent layers aren't redundant.
    total_range = max(centres[0] - centres[-1], 0.10)
    min_sep = max(total_range * 0.08, 0.03)
    for i in range(1, len(centres)):
        if centres[i - 1] - centres[i] < min_sep:
            centres[i] = centres[i - 1] - min_sep

    return centres


def _build_soft_masks(
    depth_map: np.ndarray,
    centres: list[float],
    weight_map: np.ndarray | None = None,
) -> np.ndarray:
    """Build soft Gaussian alpha masks from depth band centres.

    Each pixel is assigned proportionally to the nearest band centre.
    Optional *weight_map* zeroes out regions (e.g. subject area for portrait bg).
    """
    n = len(centres)

    # Minimum sigma scales with the total depth range so that even
    # tightly clustered centres (e.g. bimodal ground mode) produce
    # overlapping masks with meaningful coverage in every band.
    total_range = max(centres[0] - centres[-1], 0.10) if n > 1 else 0.20
    min_sigma = max(total_range / (n * 2.5), 0.04)

    # Sigma = half the distance to the nearest neighbour (clamped to min_sigma).
    sigmas: list[float] = []
    for i, c in enumerate(centres):
        dists = []
        if i > 0:
            dists.append(abs(centres[i - 1] - c))
        if i < n - 1:
            dists.append(abs(centres[i + 1] - c))
        nearest = min(dists) if dists else 0.20
        sigmas.append(max(nearest * 0.55, min_sigma))

    bands = [_gaussian_mask(depth_map, c, s) for c, s in zip(centres, sigmas)]
    stack = np.stack(bands, axis=0)

    # Normalize so bands sum to 1.0 at every pixel.
    denom = np.maximum(stack.sum(axis=0), 1e-6)
    stack /= denom[None, ...]

    # Light spatial smoothing to prevent noise-driven speckle.
    for i in range(n):
        stack[i] = cv2.GaussianBlur(stack[i], (0, 0), sigmaX=3.0, sigmaY=3.0)

    # Re-normalize after smoothing.
    denom = np.maximum(stack.sum(axis=0), 1e-6)
    stack /= denom[None, ...]

    # Apply weight map last (e.g. zero out subject area for background layers).
    if weight_map is not None:
        stack *= weight_map[None, ...]

    return stack.astype(np.float32)


# ---------------------------------------------------------------------------
# Scene classification
# ---------------------------------------------------------------------------


def classify_scene(
    subject_alpha: np.ndarray,
    depth_map: np.ndarray,
    is_landscape: bool = False,
) -> SceneType:
    """Decide whether a scene is PORTRAIT (compact subject) or VISTA (terrain).

    PORTRAIT mode extracts the subject, inpaints behind it, and depth-splits
    the background.  VISTA mode ignores the subject mask entirely and
    decomposes the *full original image* into depth-ordered layers — no
    inpainting, no hallucinated content.

    Decision tree:
    1. SubjectRunner already flagged landscape → VISTA
    2. Coverage <3% or >55% → VISTA (no compact subject)
    3. Subject has high internal depth variance → VISTA (terrain geometry)
    4. Subject mask touches 3+ image edges → VISTA (scene-filling, not object)
    5. Otherwise → PORTRAIT
    """
    if is_landscape:
        return SceneType.VISTA

    mask_binary = subject_alpha > 0.3
    coverage = float(mask_binary.mean())

    # Too little or too much coverage → no compact subject
    if coverage < 0.03 or coverage > 0.55:
        return SceneType.VISTA

    # Check whether the "subject" has consistent depth (object) or
    # high depth variance (terrain / distributed geometry).
    subject_pixels = depth_map[mask_binary]
    if subject_pixels.size < 100:
        return SceneType.VISTA

    depth_std = float(subject_pixels.std())
    p10 = float(np.percentile(subject_pixels, 10))
    p90 = float(np.percentile(subject_pixels, 90))
    depth_range = p90 - p10

    if depth_std > 0.12 or depth_range > 0.35:
        return SceneType.VISTA

    # Edge-touching heuristic: if the mask extends to 3+ image borders,
    # it's likely scene geometry (canyon walls, terrain) not a compact object.
    h, w = subject_alpha.shape
    margin = max(3, min(h, w) // 100)
    edges_touched = (
        int(mask_binary[:margin, :].any())  # top
        + int(mask_binary[-margin:, :].any())  # bottom
        + int(mask_binary[:, :margin].any())  # left
        + int(mask_binary[:, -margin:].any())  # right
    )
    if edges_touched >= 3:
        return SceneType.VISTA

    return SceneType.PORTRAIT


# ---------------------------------------------------------------------------
# VISTA mode — pure depth decomposition via K-means
# ---------------------------------------------------------------------------


def _build_vista_masks(depth_map: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose the full image into 4 depth-ordered layers for vista scenes.

    Uses K-means clustering on the depth map to find natural depth centres.
    K-means inherently handles bimodal (sky + ground), continuous gradients,
    and everything in between — no histogram valleys, no bimodal detection,
    no IQR heuristics.

    K-means runs on *raw* depth (preserves sub-structure within modes).
    Masks are built from *smoothed* depth (clean spatial boundaries).

    No subject extraction.  No inpainting.
    """
    smooth = edge_preserving_smooth(depth_map, sigma_space=20.0, sigma_color=0.12)

    # Cluster on RAW depth to capture natural distribution sub-structure.
    raw_flat = depth_map.ravel()
    valid = raw_flat[raw_flat > 0.01]
    if valid.size < 256:
        valid = raw_flat

    centres = _find_depth_centres(valid, n=4)
    # Build masks on SMOOTHED depth for clean spatial boundaries.
    stack = _build_soft_masks(smooth, centres)

    return {
        LAYER_NAMES[0]: stack[0],
        LAYER_NAMES[1]: stack[1],
        LAYER_NAMES[2]: stack[2],
        LAYER_NAMES[3]: stack[3],
        LAYER_NAMES[4]: np.zeros_like(depth_map, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# PORTRAIT mode — subject as hero + background depth split via K-means
# ---------------------------------------------------------------------------


def _build_front_fx(
    clean_subject: np.ndarray,
    depth_map: np.ndarray,
) -> np.ndarray:
    """Thin near-side rim around the subject for bokeh / bloom FX."""
    binary = (clean_subject > 0.24).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(clean_subject, dtype=np.float32)

    h, w = clean_subject.shape
    shell_size = max(3, ((min(h, w) // 220) * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shell_size, shell_size))
    eroded = cv2.erode(binary, kernel)
    dilated = cv2.dilate(binary, kernel)
    shell = (dilated.astype(np.float32) - eroded.astype(np.float32)).clip(0.0, 1.0)

    subject_depth = depth_map[binary > 0]
    near_cut = float(np.percentile(subject_depth, 38)) if subject_depth.size else 0.28
    near_bias = np.clip((near_cut + 0.16 - depth_map) / 0.16, 0.0, 1.0)
    return np.clip(shell * clean_subject * near_bias * 0.7, 0.0, 1.0)


def _build_portrait_masks(
    depth_map: np.ndarray,
    subject_alpha: np.ndarray,
) -> dict[str, np.ndarray]:
    """Subject = hero layer.  Background split into 3 depth bands via K-means.

    * Layers 0-2: background depth bands (subject area masked out).
    * Layer 3 (hero_fg): the detected subject.
    * Layer 4 (front_fx): thin near-side rim for bokeh / bloom.
    """
    clean_subject = _clean_subject_alpha(subject_alpha)
    bg_weight = 1.0 - clean_subject

    # K-means on background depth pixels for 3 background bands.
    bg_depth = depth_map[bg_weight > 0.08]
    if bg_depth.size < 128:
        bg_depth = depth_map.ravel()

    centres = _find_depth_centres(bg_depth, n=3)
    bg_stack = _build_soft_masks(depth_map, centres, weight_map=bg_weight)

    # Hero foreground — subject matte.
    hero_mask = clean_subject.copy()

    # Front FX — near-side rim for bokeh.
    fx_mask = _build_front_fx(clean_subject, depth_map)
    hero_mask = np.clip(hero_mask - fx_mask * 0.35, 0.0, 1.0)

    return {
        LAYER_NAMES[0]: bg_stack[0],
        LAYER_NAMES[1]: bg_stack[1],
        LAYER_NAMES[2]: bg_stack[2],
        LAYER_NAMES[3]: hero_mask,
        LAYER_NAMES[4]: fx_mask,
    }


def plan_layers(
    wallpaper_id: str,
    config: PipelineConfig,
    depth_map: np.ndarray | None = None,
    subject_alpha: np.ndarray | None = None,
    is_landscape: bool = False,
) -> PlannedScene:
    """Plan the five-layer decomposition for a wallpaper.

    Focus-driven approach:
    - Find the focus (BiRefNet subject or depth-derived).
    - Separate focus from background.
    - Split background by depth using K-means clustering.
    - Create 3D parallax through differential layer motion.
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

    scene_type = SceneType.PORTRAIT
    masks: dict[str, np.ndarray] = {}

    if depth_map is not None and subject_alpha is not None:
        scene_type = classify_scene(subject_alpha, depth_map, is_landscape)

        if scene_type == SceneType.VISTA:
            masks = _build_vista_masks(depth_map)
        else:
            masks = _build_portrait_masks(depth_map, subject_alpha)

    return PlannedScene(
        layers=layers,
        safe_clock_rect=config.safe_clock_rect,
        scene_type=scene_type,
        layer_masks=masks,
        depth_breaks=[],
    )
