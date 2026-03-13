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
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.warnings) == 0

    def summary_line(self) -> str:
        tag = "PASS" if self.passed else "WARN"
        return (
            f"[{tag}] depth_sep={self.depth_separation:.2f} "
            f"mask_clean={self.mask_cleanliness:.2f} "
            f"clock_read={self.clock_readability:.2f}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "depth_separation": round(self.depth_separation, 4),
            "mask_cleanliness": round(self.mask_cleanliness, 4),
            "clock_readability": round(self.clock_readability, 4),
            "warnings": self.warnings,
            "passed": self.passed,
        }


def score_scene(
    depth_map: np.ndarray,
    subject_alpha: np.ndarray,
    safe_clock_rect: tuple[float, float, float, float],
    thresholds: QualityThresholds | None = None,
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

    # --- 3. Clock readability: clearance in the safe rect ---
    h, w = subject_alpha.shape
    l, t, r, b = safe_clock_rect
    clock_region = subject_alpha[int(t * h) : int(b * h), int(l * w) : int(r * w)]
    if clock_region.size > 0:
        clock_score = 1.0 - float((clock_region > 0.3).mean())
    else:
        clock_score = 1.0
    if clock_score < thresholds.min_clock_clearance:
        warnings.append(f"Clock zone obstructed: {clock_score:.2f}")

    return QualityReport(
        depth_separation=depth_sep,
        mask_cleanliness=cleanliness,
        clock_readability=clock_score,
        warnings=warnings,
    )
