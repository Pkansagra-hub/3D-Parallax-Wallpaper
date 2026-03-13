from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from parallaxgen.config import QualityThresholds


@dataclass(slots=True)
class QualityReport:
    depth_separation: float
    mask_cleanliness: float
    clock_readability: float
    plane_cohesion: float = 1.0
    foreground_cohesion: float = 1.0
    front_shell_balance: float = 1.0
    layer_differentiation: float = 1.0
    scene_type: str = "portrait"
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.warnings) == 0

    def summary_line(self) -> str:
        tag = "PASS" if self.passed else "WARN"
        return (
            f"[{tag}] depth_sep={self.depth_separation:.2f} "
            f"mask_clean={self.mask_cleanliness:.2f} "
            f"clock_read={self.clock_readability:.2f} "
            f"plane_cohesion={self.plane_cohesion:.2f} "
            f"fg_cohesion={self.foreground_cohesion:.2f} "
            f"layer_diff={self.layer_differentiation:.2f} "
            f"mode={self.scene_type}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "depth_separation": round(self.depth_separation, 4),
            "mask_cleanliness": round(self.mask_cleanliness, 4),
            "clock_readability": round(self.clock_readability, 4),
            "plane_cohesion": round(self.plane_cohesion, 4),
            "foreground_cohesion": round(self.foreground_cohesion, 4),
            "front_shell_balance": round(self.front_shell_balance, 4),
            "layer_differentiation": round(self.layer_differentiation, 4),
            "scene_type": self.scene_type,
            "warnings": self.warnings,
            "passed": self.passed,
        }


def _largest_component_ratio(mask: np.ndarray, threshold: float = 0.2) -> float:
    binary = (mask > threshold).astype(np.uint8)
    total = int(binary.sum())
    if total == 0:
        return 1.0

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 1.0

    largest = int(stats[1:, cv2.CC_STAT_AREA].max(initial=0))
    return float(largest / max(total, 1))


def score_scene(
    depth_map: np.ndarray,
    subject_alpha: np.ndarray,
    safe_clock_rect: tuple[float, float, float, float],
    thresholds: QualityThresholds | None = None,
    layer_masks: dict[str, np.ndarray] | None = None,
    scene_type: str = "portrait",
) -> QualityReport:
    """Score a processed scene on three quality axes.

    Returns a :class:`QualityReport` with per-axis scores in ``[0, 1]`` and
    human-readable warnings for anything below threshold.
    """
    thresholds = thresholds or QualityThresholds()
    warnings: list[str] = []

    # --- 1. Depth separation: how much of the [0,1] range is used? ---
    p5, p95 = float(np.percentile(depth_map, 5)), float(np.percentile(depth_map, 95))
    depth_sep = p95 - p5
    if depth_sep < thresholds.min_depth_separation:
        warnings.append(f"Weak depth separation: {depth_sep:.2f}")

    # --- 2. Mask cleanliness: convex-hull solidity ---
    # Solidity = contour area / convex hull area.  Values close to 1.0 mean
    # a compact subject; complex but legitimate silhouettes (hair, arms)
    # score ~0.6-0.8 instead of being penalised by perimeter-based metrics.
    subj_u8 = (subject_alpha > 0.3).astype(np.uint8) * 255
    contours, _ = cv2.findContours(subj_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        total_area = sum(cv2.contourArea(c) for c in contours)
        total_hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)
        cleanliness = total_area / max(total_hull_area, 1.0)
    else:
        cleanliness = 0.5
    if cleanliness < thresholds.min_subject_coverage:
        warnings.append(f"Noisy subject mask edges: {cleanliness:.2f}")

    # --- 3. Clock readability: partial occlusion is valid, but a mostly hidden
    # clock is not.  The Apple-style effect expects foreground overlap.
    h, w = subject_alpha.shape
    l, t, r, b = safe_clock_rect
    clock_region = subject_alpha[int(t * h) : int(b * h), int(l * w) : int(r * w)]
    if clock_region.size > 0:
        visible_ratio = 1.0 - float((clock_region > 0.3).mean())
    else:
        visible_ratio = 1.0

    min_visible_ratio = max(0.18, thresholds.min_clock_clearance * 0.35)
    clock_score = float(np.clip(visible_ratio / min_visible_ratio, 0.0, 1.0))
    if visible_ratio < min_visible_ratio:
        warnings.append(f"Clock mostly hidden: {visible_ratio:.2f} visible")

    plane_cohesion = 1.0
    foreground_cohesion = 1.0
    front_shell_balance = 1.0
    layer_differentiation = 1.0
    if layer_masks:
        background_scores = [
            _largest_component_ratio(layer_masks[name])
            for name in ("layer_0_far_bg", "layer_1_deep_mid", "layer_2_near_mid")
            if name in layer_masks and float(layer_masks[name].mean()) > 0.015
        ]
        if background_scores:
            plane_cohesion = float(np.mean(background_scores))
            if plane_cohesion < 0.32:
                warnings.append(
                    f"Background planes too fragmented: {plane_cohesion:.2f}"
                )

        hero_mask = layer_masks.get("layer_3_hero_fg")
        if hero_mask is not None and float(hero_mask.mean()) > 0.005:
            foreground_cohesion = _largest_component_ratio(hero_mask)
            if foreground_cohesion < 0.55:
                warnings.append(f"Hero plane too fragmented: {foreground_cohesion:.2f}")

        fx_mask = layer_masks.get("layer_4_front_fx")
        if fx_mask is not None:
            hero_area = float(hero_mask.mean()) if hero_mask is not None else 0.0
            fx_area = float(fx_mask.mean())
            if hero_area > 0.01:
                fx_ratio = fx_area / max(hero_area, 1e-5)
                front_shell_balance = float(
                    np.clip(1.0 - max(fx_ratio - 0.22, 0.0), 0.0, 1.0)
                )
                if fx_ratio > 0.35:
                    warnings.append(f"Front shell too dominant: {fx_ratio:.2f}")

        # --- Visual mass: each mid layer should contribute meaningful content ---
        for name in ("layer_1_deep_mid", "layer_2_near_mid", "layer_3_hero_fg"):
            mask = layer_masks.get(name)
            if mask is not None:
                visual_mass = float((mask > 0.15).mean())
                if visual_mass < 0.03:
                    warnings.append(
                        f"{name} has negligible content ({visual_mass:.1%})"
                    )

        # --- Layer differentiation: pairwise check for redundant layers ---
        # Two layers with >85% alpha overlap produce zero parallax between
        # them — a user-visible quality problem.
        active_names = [
            n
            for n in (
                "layer_0_far_bg",
                "layer_1_deep_mid",
                "layer_2_near_mid",
                "layer_3_hero_fg",
            )
            if n in layer_masks and float((layer_masks[n] > 0.15).mean()) > 0.02
        ]
        pair_scores: list[float] = []
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                a = (layer_masks[active_names[i]] > 0.15).astype(np.float32)
                b = (layer_masks[active_names[j]] > 0.15).astype(np.float32)
                intersection = float((a * b).sum())
                union = float(np.maximum(a, b).sum())
                iou = intersection / max(union, 1.0)
                pair_scores.append(1.0 - iou)  # 1.0 = fully distinct, 0.0 = identical
        if pair_scores:
            layer_differentiation = float(np.mean(pair_scores))
            worst_pair = 1.0 - min(pair_scores)  # highest IoU
            if worst_pair > 0.85:
                warnings.append(
                    f"Redundant layers detected (worst IoU={worst_pair:.2f})"
                )

    return QualityReport(
        depth_separation=depth_sep,
        mask_cleanliness=cleanliness,
        clock_readability=clock_score,
        plane_cohesion=plane_cohesion,
        foreground_cohesion=foreground_cohesion,
        front_shell_balance=front_shell_balance,
        layer_differentiation=layer_differentiation,
        scene_type=scene_type,
        warnings=warnings,
    )
