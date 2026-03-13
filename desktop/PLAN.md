# Desktop Plan — End-to-End Implementation

This document is the single source of truth for every component in the ParallaxGen desktop pipeline. Every module has exact model names, pip packages, code-level implementation details, and acceptance criteria. No stubs. No "choose later". Every decision is made here.

---

## Current State (Honest Assessment)

**Done — Milestone 1 (Foundations):**

- Package schema frozen in `models.py` (S26 Ultra 1440×3120, v2 format)
- Config layer in `config.py` (PipelineConfig, QualityThresholds)
- CLI wired with all config options in `cli.py`
- Manifest/packer write real files in `corpus/`
- 10 sample wallpapers in `desktop/src/parallaxgen/Images/`
- 6 tests in `test_manifest.py` — all passing

**Done — Milestone 2 (Real Depth + Subject Extraction):**

- `depth_runner.py` → Depth Anything V2 Large (335M params), 0.47s warm on RTX 5070
- `depth_utils.py` → smooth_depth, edge_preserving_smooth, compute_depth_histogram_breaks
- `subject_runner.py` → BiRefNet via HuggingFace transformers, torchvision preprocessing
- `matte_refiner.py` → OpenCV morphological open/close + Gaussian edge smoothing

**Done — Milestone 3 (Scene Composition + Layer Rendering):**

- `layer_planner.py` → Depth-driven semantic planner using histogram breaks, per-layer masks
- `scene_builder.py` → Full pipeline wiring: depth → segmentation → refinement → planning → compositing
- `occlusion_planner.py` → Data-driven clock occlusion from real subject matte + safe rect finder
- `models.py` → Dynamic clock rendering metadata (font_scale, anchor, weight, safe_rect, occlusion)

**Done — Milestone 4 (Inpainting, Packaging, Preview):**

- `inpainter.py` → OpenCV Telea inpainting (cv2) with LaMa ML fallback
- `quality_scorer.py` → Three-axis scoring (depth separation, mask cleanliness, clock readability)
- `packer.py` → Corpus validation, index.json enforcement, archive integrity checks
- `preview_renderer.py` → 2×2 QA grid (original+clock, depth heatmap, subject mask, layer assignment)
- Quality scores written into `meta.json`, QA grid exported as `qa_grid.webp`
- Full pipeline tested: 23.59s end-to-end on RTX 5070, quality PASSED

**Done — Milestone 5 (Hardening + Developer Experience):**

- D-16: Structured logging + tqdm progress bars (all modules + CLI batch)
- D-17: Benchmark CLI command (`parallaxgen benchmark`)
- D-18: Setup documentation (README.md)

**Done — Milestone 6 (Visual Quality Enhancements):**

- D-19: LaMa as default inpainter (with cv2 fallback)
- D-20: Adaptive DOF blur (depth-driven per-layer blur replaces fixed blur_px)
- D-21: Far-bg AMOLED vignette + front_fx chromatic aberration
- D-22: Display P3 color pipeline (ICC-aware load, linear-light blending, P3 output)
- D-23: BiRefNet-HR 1536×1536 high-res segmentation variant
- D-24: Depth Pro model backend (Apple, metric depth, sharper boundaries)

> **Android-side enhancements (not in this milestone — see `android/DESKTOP_HANDOFF.md`):**
> Spring physics parallax, 2D vertical+horizontal parallax, non-linear depth weight curves, HDR10+ rendering.

---

## Epic: Desktop Spatial Package Pipeline MVP

**One command in, real spatial wallpaper package out.**

```bash
parallaxgen process photo.jpg --output ./corpus
```

This must produce a `.parallax`-ready folder with:

- 5 RGBA layer WebPs generated from real depth + real segmentation
- Real subject mask derived from ML inference
- Real normalized depth map from ML inference
- Inpainted background with subject holes filled
- Data-driven clock occlusion mask
- Data-driven safe clock rect
- Visual preview image with clock overlay and QA cues
- Valid `meta.json` and `index.json` matching Android loader expectations

**Done when:** You can process the 10 sample images and visually confirm convincing layer separation, clean subject extraction, and usable clock placement on all 10.

---

## Required Python Dependencies

These go in `pyproject.toml` under `[project.dependencies]` (core) and `[project.optional-dependencies] ml` (heavy ML):

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "pillow>=10.0",
    "typer>=0.12",
    "opencv-python>=4.9",
    "tqdm>=4.66",
]

[project.optional-dependencies]
ml = [
    "torch>=2.2",
    "torchvision>=0.17",
    "timm>=1.0",
    "transformers>=4.40",
    "simple-lama-inpainting>=0.1.2",
]
```

Install:

```bash
cd desktop
pip install -e ".[ml]"
```

GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1.

---

## Milestone 1: Pipeline Foundations ✅ DONE

Issues D-1 through D-4 are complete. Package contract frozen, config layer added, asset writes produce real images (from stub data), tests pass.

---

## Milestone 2: Real Depth + Real Subject Extraction

This is the critical milestone. After this, the pipeline produces real data instead of synthetic garbage.

### Issue D-5: Real Depth Model — Depth Anything V2

**File:** `desktop/src/parallaxgen/depth/depth_runner.py`

**Model:** Depth Anything V2 (Large)

- Paper: "Depth Anything V2" (2024), state-of-the-art monocular depth
- HuggingFace: `depth-anything/Depth-Anything-V2-Large`
- Pip: `transformers` (AutoModelForDepthEstimation, AutoImageProcessor)
- Parameters: 335M (Large variant — best quality-to-speed ratio for desktop)
- Output: dense relative depth map, higher values = farther away (inverted from MiDaS convention)

**Fallback Model:** MiDaS DPT-Large

- Torch Hub: `intel-isl/MiDaS`, model `DPT_Large`
- Pip: `torch`, `timm`
- Used when `--depth-model midas_dpt_large` is passed

**Implementation:**

```python
# depth_runner.py — full replacement

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

@dataclass(slots=True)
class DepthResult:
    width: int
    height: int
    depth_map: np.ndarray        # float32 [0.0, 1.0], 0=near 1=far
    model_name: str

class DepthRunner:
    def __init__(self, model_name="depth_anything_v2_large",
                 output_resolution=(1440, 3120)) -> None:
        self.model_name = model_name
        self.output_resolution = output_resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.model_name.startswith("depth_anything"):
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
            repo = "depth-anything/Depth-Anything-V2-Large"
            self._processor = AutoImageProcessor.from_pretrained(repo)
            self._model = AutoModelForDepthEstimation.from_pretrained(repo)
            self._model.to(self.device).eval()
        elif self.model_name == "midas_dpt_large":
            self._model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            self._model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._processor = midas_transforms.dpt_transform

    def infer(self, image_path: Path) -> DepthResult:
        self._load_model()
        image = Image.open(image_path).convert("RGB")
        w, h = self.output_resolution

        if self.model_name.startswith("depth_anything"):
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                predicted_depth = outputs.predicted_depth
            # Resize to output resolution
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
            # Normalize to [0, 1] — Depth Anything: higher = farther
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        elif self.model_name == "midas_dpt_large":
            import cv2
            img_np = np.array(image)
            input_batch = self._processor(img_np).to(self.device)
            with torch.no_grad():
                prediction = self._model(input_batch)
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(h, w),
                mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
            # MiDaS: higher = nearer, so invert to match our convention (0=near, 1=far)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = 1.0 - depth

        return DepthResult(width=w, height=h,
                           depth_map=depth.astype(np.float32),
                           model_name=self.model_name)
```

**Key decisions:**

- Convention: 0.0 = nearest to camera, 1.0 = farthest. This matches how layer_planner slices depth bands.
- Lazy model loading: `_load_model()` called on first `infer()` so import is fast.
- CUDA auto-detect: uses GPU when available, falls back to CPU with no config change.
- Model weights auto-download on first run via HuggingFace / torch hub cache.

**Acceptance criteria:**

- Process a landscape photo → depth map shows sky/background bright (high values), foreground objects dark (low values)
- Process a portrait photo → person is dark (near), background is bright (far)
- Depth map resolution matches `output_resolution`
- Works on CPU (slower) and CUDA (fast)

---

### Issue D-6: Depth Post-Processing

**File:** `desktop/src/parallaxgen/depth/depth_utils.py`

**Current state:** Has `normalize_depth()` only.

**Add these utilities:**

```python
import numpy as np
import cv2

def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Normalize to [0.0, 1.0] range."""
    dmin, dmax = depth_map.min(), depth_map.max()
    if dmax - dmin < 1e-8:
        return np.zeros_like(depth_map, dtype=np.float32)
    return ((depth_map - dmin) / (dmax - dmin)).astype(np.float32)

def smooth_depth(depth_map: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smooth to reduce noise while preserving large structures."""
    ksize = int(sigma * 6) | 1  # must be odd
    return cv2.GaussianBlur(depth_map, (ksize, ksize), sigma)

def edge_preserving_smooth(depth_map: np.ndarray, sigma_s: float = 30.0,
                            sigma_r: float = 0.15) -> np.ndarray:
    """Bilateral filter: smooth flat regions, preserve depth edges."""
    depth_u8 = (depth_map * 255).astype(np.uint8)
    smoothed = cv2.bilateralFilter(depth_u8, d=-1,
                                    sigmaColor=sigma_r * 255,
                                    sigmaSpace=sigma_s)
    return smoothed.astype(np.float32) / 255.0

def compute_depth_histogram_breaks(depth_map: np.ndarray,
                                    n_bins: int = 256) -> list[float]:
    """Find natural depth band boundaries using histogram valleys."""
    hist, bin_edges = np.histogram(depth_map.flatten(), bins=n_bins, range=(0, 1))
    # Smooth histogram
    kernel = np.ones(5) / 5
    smooth_hist = np.convolve(hist, kernel, mode='same')
    # Find local minima (valleys = natural layer boundaries)
    valleys = []
    for i in range(1, len(smooth_hist) - 1):
        if smooth_hist[i] < smooth_hist[i-1] and smooth_hist[i] < smooth_hist[i+1]:
            valleys.append(bin_edges[i])
    return sorted(valleys)
```

**Why:**

- `smooth_depth` reduces sensor/model noise that creates flickery layer edges
- `edge_preserving_smooth` keeps sharp depth discontinuities (mountain against sky) while smoothing flat regions
- `compute_depth_histogram_breaks` finds natural depth band boundaries instead of using fixed thresholds — this is how we stop doing `np.linspace` for layer bands

---

### Issue D-7: Real Subject Segmentation — BiRefNet

**File:** `desktop/src/parallaxgen/segment/subject_runner.py`

**Model:** BiRefNet (Bilateral Reference Network)

- Paper: "Bilateral Reference for High-Resolution Dichotomous Image Segmentation" (CAAI AIR 2024)
- HuggingFace: `ZhengPeng7/BiRefNet`
- Pip: `transformers` (AutoModelForImageSegmentation)
- State-of-the-art for salient object segmentation — clean edges on hair, fur, thin structures
- Apple uses a similar approach for their spatial photos (high-res dichotomous segmentation)

**Fallback Model:** U²-Net (lighter, no torch dependency beyond base)

- HuggingFace: `briaai/RMBG-1.4` (production-ready background removal model based on IS-Net)

**Implementation:**

```python
# subject_runner.py — full replacement

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

@dataclass(slots=True)
class SubjectMask:
    alpha: np.ndarray     # float32 [0.0, 1.0], full resolution
    bbox: tuple[float, float, float, float]  # normalized (left, top, right, bottom)

class SubjectRunner:
    def __init__(self, model_name="birefnet",
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.model_name == "birefnet":
            from transformers import AutoModelForImageSegmentation, AutoImageProcessor
            repo = "ZhengPeng7/BiRefNet"
            self._processor = AutoImageProcessor.from_pretrained(
                repo, trust_remote_code=True
            )
            self._model = AutoModelForImageSegmentation.from_pretrained(
                repo, trust_remote_code=True
            )
            self._model.to(self.device).eval()
        elif self.model_name == "rmbg":
            from transformers import AutoModelForImageSegmentation, AutoImageProcessor
            repo = "briaai/RMBG-1.4"
            self._processor = AutoImageProcessor.from_pretrained(
                repo, trust_remote_code=True
            )
            self._model = AutoModelForImageSegmentation.from_pretrained(
                repo, trust_remote_code=True
            )
            self._model.to(self.device).eval()

    def infer(self, image_path: Path, width: int, height: int) -> SubjectMask:
        self._load_model()
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # BiRefNet outputs a list of side outputs; last one is the finest
        if hasattr(outputs, "logits"):
            mask_logits = outputs.logits
        else:
            # Some model versions return tuple
            mask_logits = outputs[-1]

        # Resize to target dimensions
        mask = torch.nn.functional.interpolate(
            mask_logits.unsqueeze(0) if mask_logits.dim() == 3 else mask_logits,
            size=(height, width), mode="bilinear", align_corners=False
        )
        # Sigmoid to [0, 1] probability
        alpha = torch.sigmoid(mask).squeeze().cpu().numpy().astype(np.float32)

        # Derive bounding box from mask (normalized coordinates)
        bbox = self._compute_bbox(alpha)

        return SubjectMask(alpha=alpha, bbox=bbox)

    @staticmethod
    def _compute_bbox(alpha: np.ndarray,
                      threshold: float = 0.3) -> tuple[float, float, float, float]:
        binary = alpha > threshold
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not rows.any() or not cols.any():
            return (0.0, 0.0, 1.0, 1.0)  # fallback: full frame
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h, w = alpha.shape
        return (cmin / w, rmin / h, cmax / w, rmax / h)
```

**Key decisions:**

- BiRefNet is chosen because it produces the sharpest edges on hair, fur, leaves — exactly the hard cases
- Bounding box is computed FROM the mask, not hard-coded
- `trust_remote_code=True` is required for BiRefNet's custom model class
- Same lazy loading pattern as DepthRunner

**Acceptance criteria:**

- Portrait photo → clean alpha matte isolating the person with hair detail preserved
- Landscape with tree → tree silhouette extracted with leaf edges
- No subject → returns near-zero alpha, bbox falls back to (0,0,1,1)
- Works on both CPU and CUDA

---

### Issue D-8: Matte Refinement

**File:** `desktop/src/parallaxgen/segment/matte_refiner.py`

**Current state:** Just `np.clip(alpha, 0.0, 1.0)` — does nothing useful.

**Implementation using OpenCV morphological ops + guided filter:**

```python
import numpy as np
import cv2

def refine_alpha(alpha: np.ndarray,
                 edge_radius: int = 3,
                 smooth_sigma: float = 0.8) -> np.ndarray:
    """
    Clean up raw segmentation alpha:
    1. Remove small noise blobs (morphological open)
    2. Close small holes in the mask (morphological close)
    3. Smooth edges with Gaussian to reduce staircase artifacts
    4. Re-threshold to clean binary-ish alpha with soft edges
    """
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # Convert to uint8 for morphological ops
    mask_u8 = (alpha * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (edge_radius * 2 + 1, edge_radius * 2 + 1))

    # Remove small noise islands
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # Fill small holes inside the subject
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Smooth edges
    if smooth_sigma > 0:
        ksize = int(smooth_sigma * 6) | 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (ksize, ksize), smooth_sigma)

    return mask_u8.astype(np.float32) / 255.0
```

**Why this matters:** Raw BiRefNet output can have noisy edges, thin speckles outside the subject, or small holes inside. This cleanup makes the hero_fg layer look clean when composited.

---

## Milestone 3: Scene Composition + Layer Rendering

Now that we have real depth maps and real subject masks, we build real layers.

### Issue D-9: Semantic Layer Planner

**File:** `desktop/src/parallaxgen/compose/layer_planner.py`

**Current state:** Fixed list of 5 layers with config-driven weights. Does not use depth data at all.

**New implementation:** Planner takes depth map + subject mask and decides per-pixel layer assignment.

```python
import numpy as np
from dataclasses import dataclass
from parallaxgen.config import PipelineConfig
from parallaxgen.models import LayerSpec
from parallaxgen.depth.depth_utils import compute_depth_histogram_breaks

@dataclass(slots=True)
class PlannedScene:
    layers: list[LayerSpec]
    layer_masks: dict[str, np.ndarray]  # layer_name -> float32 mask [0,1]
    safe_clock_rect: tuple[float, float, float, float]

def plan_layers(wallpaper_id: str, config: PipelineConfig,
                depth_map: np.ndarray,
                subject_alpha: np.ndarray) -> PlannedScene:
    """
    Assign every pixel to one of 5 layers using depth + subject data.

    Strategy:
    - Layer 0 (far_bg): depth >= 0.70 AND not subject → distant sky/mountains
    - Layer 1 (deep_mid): depth 0.40-0.70 AND not subject → midground structures
    - Layer 2 (near_mid): depth 0.15-0.40 AND not subject → closer objects
    - Layer 3 (hero_fg): subject alpha > 0.3 → the main subject regardless of depth
    - Layer 4 (front_fx): thin edge details from subject boundary → partial occluders

    The depth thresholds above are defaults. If histogram analysis finds natural
    breaks in the depth distribution, those are used instead.
    """
    h, w = depth_map.shape
    subject_binary = (subject_alpha > 0.3).astype(np.float32)
    non_subject = 1.0 - subject_binary

    # Try to find natural depth breaks; fall back to fixed thresholds
    breaks = compute_depth_histogram_breaks(depth_map * non_subject)
    if len(breaks) >= 2:
        t1, t2 = breaks[0], breaks[1]  # Two strongest valleys
        # Ensure reasonable ordering
        if t1 > t2:
            t1, t2 = t2, t1
        t1 = max(0.10, min(t1, 0.35))
        t2 = max(t1 + 0.10, min(t2, 0.65))
    else:
        t1, t2 = 0.20, 0.50

    # Build per-layer masks
    far_bg_mask = (depth_map >= t2).astype(np.float32) * non_subject
    deep_mid_mask = ((depth_map >= t1) & (depth_map < t2)).astype(np.float32) * non_subject
    near_mid_mask = (depth_map < t1).astype(np.float32) * non_subject
    hero_fg_mask = np.clip(subject_alpha, 0.0, 1.0)

    # Front FX: edge pixels of the subject (dilate - erode = boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    subj_u8 = (subject_binary * 255).astype(np.uint8)
    dilated = cv2.dilate(subj_u8, kernel, iterations=1)
    eroded = cv2.erode(subj_u8, kernel, iterations=1)
    front_fx_mask = ((dilated - eroded) / 255.0).astype(np.float32)

    layer_masks = {
        "layer_0_far_bg": far_bg_mask,
        "layer_1_deep_mid": deep_mid_mask,
        "layer_2_near_mid": near_mid_mask,
        "layer_3_hero_fg": hero_fg_mask,
        "layer_4_front_fx": front_fx_mask,
    }

    layers = [
        LayerSpec(name=name,
                  asset_path=f"{wallpaper_id}/{name}.webp",
                  weight=config.depth_weights[i],
                  blur_px=config.blur_px[i])
        for i, name in enumerate(layer_masks.keys())
    ]

    return PlannedScene(layers=layers, layer_masks=layer_masks,
                        safe_clock_rect=config.safe_clock_rect)
```

**Key change:** `PlannedScene` now carries `layer_masks` — actual numpy arrays that scene_builder uses to composite each layer. The planner uses depth histogram breaks instead of hard-coded thresholds.

---

### Issue D-10: Real Layer Compositing + Export

**File:** `desktop/src/parallaxgen/compose/scene_builder.py`

**What changes:** `_render_assets()` receives `layer_masks` from the planner instead of computing its own crude masks. The far_bg layer gets inpainted (see D-13).

Updated `build_scene_package()` flow:

1. `DepthRunner.infer()` → real depth map
2. `SubjectRunner.infer()` → real subject alpha
3. `refine_alpha()` → cleaned alpha with morphological ops
4. `plan_layers(depth_map, subject_alpha)` → `PlannedScene` with `layer_masks`
5. `inpaint_background()` → fill subject hole in background layer
6. `_render_assets(layer_masks)` → 9 WebP files from real masks
7. `build_clock_occlusion_mask(subject_alpha)` → data-driven clock mask
8. Preview composited with clock overlay

Each layer WebP is: original image pixels × layer alpha mask → RGBA WebP.

**Acceptance criteria:**

- Layer 0 (far_bg) shows only distant background with subject area filled via inpainting
- Layer 3 (hero_fg) shows only the subject with transparent background
- Layers don't overlap (except front_fx edge fringe)
- All 9 assets are valid non-zero WebP files

---

### Issue D-11: Data-Driven Clock Composition

**File:** `desktop/src/parallaxgen/compose/occlusion_planner.py`

**Current state:** Hard-coded rectangle. Useless.

**New implementation:**

```python
import numpy as np
import cv2

def build_clock_occlusion_mask(subject_alpha: np.ndarray,
                                threshold: float = 0.3) -> np.ndarray:
    """
    Clock occlusion mask: where the subject overlaps the top portion
    of the frame (where the clock typically sits).
    White pixels = subject occludes the clock there.
    """
    h, w = subject_alpha.shape
    # Clock region is roughly the top 35% of the frame
    clock_zone = np.zeros((h, w), dtype=np.float32)
    clock_bottom = int(h * 0.35)
    clock_zone[:clock_bottom, :] = 1.0

    # Intersection: where subject overlaps the clock zone
    mask = (subject_alpha > threshold).astype(np.float32) * clock_zone

    # Slight dilation so the clock avoids tight edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    return mask_u8

def compute_safe_clock_rect(subject_alpha: np.ndarray,
                             preferred_rect: tuple[float, float, float, float],
                             threshold: float = 0.3
                             ) -> tuple[float, float, float, float]:
    """
    Find the largest unobstructed rectangle in the top portion for clock placement.
    Falls back to preferred_rect if the scene is mostly clear.
    """
    h, w = subject_alpha.shape
    # Check how much of the preferred rect is occluded
    l, t, r, b = preferred_rect
    region = subject_alpha[int(t*h):int(b*h), int(l*w):int(r*w)]
    occlusion_ratio = (region > threshold).mean()

    if occlusion_ratio < 0.15:
        return preferred_rect  # Good enough, subject doesn't block it

    # Subject blocks the preferred zone — find clearest horizontal band
    # in the top 35% of the image
    top_region = subject_alpha[:int(h * 0.35), :]
    row_occlusion = (top_region > threshold).mean(axis=1)

    # Find the tallest contiguous run of low-occlusion rows
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

    if best_len < int(h * 0.05):
        # No good zone found — return preferred and flag for quality warning
        return preferred_rect

    return (0.10, best_start / h, 0.90, (best_start + best_len) / h)
```

**Key change:** Both the occlusion mask AND the safe clock rect are now derived from the actual subject mask. If a person's head pokes into the top of the frame, the clock knows to shift down or flag a quality warning.

---

### Issue D-12: Scene Quality Scoring

**File:** `desktop/src/parallaxgen/compose/quality_scorer.py` (NEW)

Scores each processed image on three axes:

```python
import numpy as np

@dataclass
class QualityReport:
    depth_separation_score: float    # 0-1, higher = better depth range
    mask_cleanliness_score: float    # 0-1, higher = smoother edges
    clock_readability_score: float   # 0-1, higher = more clear space for clock
    warnings: list[str]
    passed: bool

def score_scene(depth_map: np.ndarray,
                subject_alpha: np.ndarray,
                safe_clock_rect: tuple[float, float, float, float],
                thresholds: QualityThresholds) -> QualityReport:

    warnings = []

    # 1. Depth separation: how much of [0,1] range is actually used?
    p5, p95 = np.percentile(depth_map, [5, 95])
    depth_sep = p95 - p5  # Good images have wide depth range
    if depth_sep < thresholds.min_depth_separation:
        warnings.append(f"Weak depth separation: {depth_sep:.2f}")

    # 2. Mask cleanliness: ratio of edge pixels to total subject pixels
    # Noisy masks have high edge-to-area ratio
    subj_binary = (subject_alpha > 0.3).astype(np.uint8)
    contours, _ = cv2.findContours(subj_binary * 255, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        area = max(subj_binary.sum(), 1)
        # Lower ratio = cleaner mask. Normalize to 0-1 score.
        cleanliness = 1.0 - min(perimeter / (area ** 0.5) / 10.0, 1.0)
    else:
        cleanliness = 0.5
    if cleanliness < thresholds.min_mask_cleanliness:
        warnings.append(f"Noisy subject mask edges: {cleanliness:.2f}")

    # 3. Clock readability: how clear is the clock zone?
    h, w = subject_alpha.shape
    l, t, r, b = safe_clock_rect
    clock_region = subject_alpha[int(t*h):int(b*h), int(l*w):int(r*w)]
    clock_score = 1.0 - (clock_region > 0.3).mean()
    if clock_score < thresholds.min_clock_readability:
        warnings.append(f"Clock zone obstructed: {clock_score:.2f}")

    passed = len(warnings) == 0

    return QualityReport(
        depth_separation_score=depth_sep,
        mask_cleanliness_score=cleanliness,
        clock_readability_score=clock_score,
        warnings=warnings,
        passed=passed,
    )
```

Quality scores are written into `meta.json` and printed in CLI output. Warnings don't block processing but are visible.

---

## Milestone 4: Inpainting, Packaging, Preview

### Issue D-13: Real Inpainting — LaMa

**File:** `desktop/src/parallaxgen/inpaint/inpainter.py`

**Current state:** Empty dataclass. Does nothing.

**Model:** LaMa (Large Mask Inpainting)

- Pip: `simple-lama-inpainting` (wraps LaMa with a simple API)
- Runs on CPU (acceptable for v1, background layers only)
- Alternative: OpenCV Navier-Stokes inpainting (`cv2.inpaint`) as lightweight fallback

**Implementation:**

```python
import numpy as np
from PIL import Image
from pathlib import Path

def inpaint_background(background_image: Image.Image,
                       subject_alpha: np.ndarray,
                       method: str = "lama") -> Image.Image:
    """
    Fill the hole left by the subject in the background layer.

    Args:
        background_image: The full source image
        subject_alpha: Subject mask (float32 0-1), areas > 0.3 need filling
        method: "lama" for ML inpainting, "cv2" for fast OpenCV fallback
    Returns:
        Inpainted background image with subject area filled
    """
    # Create binary inpaint mask: where subject was
    inpaint_mask = (subject_alpha > 0.3).astype(np.uint8) * 255

    # Dilate mask slightly to cover edge artifacts
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)

    if method == "lama":
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        result = lama(background_image, Image.fromarray(inpaint_mask))
        return result

    elif method == "cv2":
        img_np = np.array(background_image.convert("RGB"))
        result = cv2.inpaint(img_np, inpaint_mask,
                              inpaintRadius=7,
                              flags=cv2.INPAINT_TELEA)
        return Image.fromarray(result)

    return background_image  # fallback: no inpainting
```

**Where it's used:** In `scene_builder.py`, after extracting the subject, the far_bg layer gets inpainted so you don't see a ghost silhouette of the subject in the background when parallax shifts the layers.

**Acceptance criteria:**

- Process a portrait → background layer has the person filled in with plausible surrounding content
- Process a landscape with tree → tree area in background filled with sky/ground continuation
- `--inpaint-method cv2` works as fast CPU fallback

---

### Issue D-14: Finalize Package Writer

**File:** `desktop/src/parallaxgen/corpus/manifest.py` + `packer.py`

**Current state:** Already mostly correct. Writes real bytes, validates assets.

**Remaining work:**

- Add quality scores to `meta.json` output
- Add `config_snapshot` field to `meta.json` for reproducibility
- Validate that all 9 required assets are present and > 1KB
- `packer.py` adds `index.json` at archive root
- Add `--validate` flag to `inspect` command that checks archive integrity

---

### Issue D-15: Visual Preview Generation

**File:** `desktop/src/parallaxgen/preview/preview_renderer.py`

**Current state:** Returns a JSON dict. Useless for visual QA.

**New implementation:**

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from parallaxgen.utils.image_io import encode_webp

def render_preview_image(
    base_image: Image.Image,
    layer_masks: dict[str, np.ndarray],
    safe_clock_rect: tuple[float, float, float, float],
    subject_alpha: np.ndarray,
    depth_map: np.ndarray,
    output_path: Path | None = None,
) -> bytes:
    """
    Generate a 2×2 grid preview:
    - Top-left: original image with clock overlay
    - Top-right: depth map colorized
    - Bottom-left: subject mask
    - Bottom-right: layer stack composite with colored tints
    """
    w, h = base_image.size
    cell_w, cell_h = w // 2, h // 2
    canvas = Image.new("RGB", (w, h))

    # Top-left: original with clock rect
    tl = base_image.copy().resize((cell_w, cell_h))
    draw = ImageDraw.Draw(tl)
    l, t, r, b = safe_clock_rect
    draw.rectangle((l*cell_w, t*cell_h, r*cell_w, b*cell_h),
                    outline=(0, 255, 0), width=3)
    draw.text((int(l*cell_w)+8, int(t*cell_h)+8), "12:40",
              fill=(255, 255, 255))
    canvas.paste(tl, (0, 0))

    # Top-right: depth map as heatmap
    depth_u8 = (depth_map * 255).astype(np.uint8)
    import cv2
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    tr = Image.fromarray(depth_color).resize((cell_w, cell_h))
    canvas.paste(tr, (cell_w, 0))

    # Bottom-left: subject mask
    mask_u8 = (subject_alpha * 255).astype(np.uint8)
    bl = Image.fromarray(mask_u8, mode="L").convert("RGB").resize((cell_w, cell_h))
    canvas.paste(bl, (0, cell_h))

    # Bottom-right: color-coded layer stack
    colors = [(30, 60, 120), (60, 120, 60), (120, 120, 30),
              (200, 60, 60), (200, 200, 60)]
    layer_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (name, mask) in enumerate(layer_masks.items()):
        for c in range(3):
            layer_vis[:, :, c] += (mask * colors[i][c]).astype(np.uint8)
    br = Image.fromarray(layer_vis).resize((cell_w, cell_h))
    canvas.paste(br, (cell_w, cell_h))

    webp_bytes = encode_webp(canvas)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(webp_bytes)
    return webp_bytes
```

**Result:** A single preview image showing all the data: original with clock zone, depth heatmap, subject mask, and color-coded layer assignment. Immediately tells you if the ML models did their job.

---

## Milestone 5: Hardening + Developer Experience

### Issue D-16: Structured Logging + Progress

**Implementation:** Use `tqdm` progress bars for batch mode. Use Python `logging` module with structured format:

```
[parallaxgen] INFO  Loading depth model: depth_anything_v2_large (cuda)
[parallaxgen] INFO  Processing: ocean_sunset.png
[parallaxgen] INFO    Stage 1/6: Depth inference .............. 2.3s
[parallaxgen] INFO    Stage 2/6: Subject segmentation ......... 1.8s
[parallaxgen] INFO    Stage 3/6: Matte refinement ............. 0.1s
[parallaxgen] INFO    Stage 4/6: Layer planning ............... 0.2s
[parallaxgen] INFO    Stage 5/6: Inpainting ................... 4.1s
[parallaxgen] INFO    Stage 6/6: Asset export ................. 0.8s
[parallaxgen] INFO  Quality: depth_sep=0.82 mask_clean=0.91 clock_read=0.95 ✓
[parallaxgen] INFO  Written: corpus/ocean_sunset/
[parallaxgen] INFO  Processing: 1/10 complete
```

### Issue D-17: Benchmark + Sample Fixtures

```bash
parallaxgen benchmark ./Images --device cuda --runs 3
```

Output:

```
Image                    Depth(s)  Segment(s)  Inpaint(s)  Total(s)
ocean_sunset.png         2.31      1.82        4.12        9.41
mountain_lake.png        2.28      1.76        3.98        9.18
...
Average                  2.30      1.79        4.05        9.30
Device: NVIDIA RTX 5070, CUDA 12.1
```

### Issue D-18: Setup Documentation

Create `desktop/README.md` with:

- System requirements (Python 3.10+, 8GB+ RAM, NVIDIA GPU recommended)
- Install commands for CPU and CUDA paths
- First run instructions (models auto-download, ~2GB)
- CLI reference for all 5 commands
- Package format reference for Android team
- Troubleshooting: OOM, CUDA not found, model download failures

---

## End-to-End Pipeline Flow (Final)

```
INPUT: photo.jpg (any resolution)
  │
  ├─ [1] DepthRunner.infer()
  │      Model: Depth Anything V2 Large (335M params)
  │      Input: RGB image
  │      Output: float32 depth map [0=near, 1=far], resized to 1440×3120
  │      Device: CUDA preferred, CPU fallback
  │      Time: ~2s GPU, ~15s CPU
  │
  ├─ [2] SubjectRunner.infer()
  │      Model: BiRefNet (salient object segmentation)
  │      Input: RGB image
  │      Output: float32 alpha matte [0,1] at 1440×3120 + normalized bbox
  │      Device: CUDA preferred, CPU fallback
  │      Time: ~2s GPU, ~12s CPU
  │
  ├─ [3] refine_alpha()
  │      Technique: morphological open/close + Gaussian edge smoothing
  │      Input: raw alpha from BiRefNet
  │      Output: cleaned alpha, noise removed, edges smoothed
  │      Time: <0.1s
  │
  ├─ [4] plan_layers(depth_map, subject_alpha)
  │      Technique: depth histogram breaks + subject mask overlay
  │      Input: depth map + refined subject alpha
  │      Output: PlannedScene with 5 per-pixel layer masks
  │      Logic: far_bg / deep_mid / near_mid split by depth valleys,
  │             hero_fg from subject alpha, front_fx from subject edge
  │      Time: <0.2s
  │
  ├─ [5] inpaint_background()
  │      Model: LaMa (via simple-lama-inpainting)
  │      Input: source image + dilated subject mask
  │      Output: background image with subject area filled
  │      Device: CPU (v1)
  │      Time: ~4s
  │
  ├─ [6] build_clock_occlusion_mask(subject_alpha)
  │      Technique: subject × top-35%-zone intersection + dilation
  │      Input: refined subject alpha
  │      Output: uint8 mask showing where subject occludes clock
  │
  ├─ [7] compute_safe_clock_rect(subject_alpha)
  │      Technique: find tallest clear horizontal band in top 35%
  │      Input: refined subject alpha + preferred rect from config
  │      Output: normalized (l, t, r, b) rect for clock placement
  │
  ├─ [8] score_scene()
  │      Metrics: depth_separation, mask_cleanliness, clock_readability
  │      Output: QualityReport with pass/warn + numeric scores
  │
  ├─ [9] _render_assets()
  │      For each layer: source_pixels × layer_mask → RGBA WebP
  │      far_bg uses inpainted source (no ghost subject)
  │      9 output files: 5 layers + depth_map + subject_mask +
  │                      clock_occlusion_mask + preview
  │      Format: WebP, quality 90, 1440×3120
  │
  └─ [10] write_package_files()
         Output directory structure:
           corpus/
             index.json
             ocean_sunset/
               meta.json
               layer_0_far_bg.webp
               layer_1_deep_mid.webp
               layer_2_near_mid.webp
               layer_3_hero_fg.webp
               layer_4_front_fx.webp
               clock_occlusion_mask.webp
               subject_mask.webp
               depth_map.webp
               preview.webp

FINAL OUTPUT: corpus/ folder ready for Android app consumption
              OR: .parallax ZIP archive via `parallaxgen pack`
```

---

## Implementation Order

| Order | Issue | File(s) | What | Status |
|-------|-------|---------|------|--------|
| 1 | D-1 | `models.py` | Freeze package contract | ✅ Done |
| 2 | D-2 | `config.py`, `cli.py` | Config layer | ✅ Done |
| 3 | D-3 | `scene_builder.py`, `image_io.py` | Non-empty asset writes | ✅ Done |
| 4 | D-4 | `tests/` | Test baseline | ✅ Done |
| 5 | D-5 | `depth/depth_runner.py` | Real depth: Depth Anything V2 | ✅ Done |
| 6 | D-6 | `depth/depth_utils.py` | Depth post-processing | ✅ Done |
| 7 | D-7 | `segment/subject_runner.py` | Real segmentation: BiRefNet | ✅ Done |
| 8 | D-8 | `segment/matte_refiner.py` | Morphological alpha cleanup | ✅ Done |
| 9 | D-9 | `compose/layer_planner.py` | Depth-driven semantic planner | ✅ Done |
| 10 | D-10 | `compose/scene_builder.py` | Wire real masks into compositing | ✅ Done |
| 11 | D-11 | `compose/occlusion_planner.py` | Data-driven clock analysis | ✅ Done |
| 12 | D-12 | `compose/quality_scorer.py` | Quality scoring (NEW file) | ✅ Done |
| 13 | D-13 | `inpaint/inpainter.py` | OpenCV Telea + LaMa fallback | ✅ Done |
| 14 | D-14 | `corpus/manifest.py`, `packer.py` | Package writer + validation | ✅ Done |
| 15 | D-15 | `preview/preview_renderer.py` | Visual 2×2 QA preview grid | ✅ Done |
| 16 | D-16 | `cli.py` + all modules | Structured logging + tqdm | ✅ Done |
| 17 | D-17 | `tests/`, `cli.py` | Benchmark script | ✅ Done |
| 18 | D-18 | `README.md` | Setup + onboarding docs | ✅ Done |
| 19 | D-19 | `inpainter.py`, `scene_builder.py` | LaMa default inpainter | ✅ Done |
| 20 | D-20 | `scene_builder.py`, `config.py` | Adaptive DOF blur | ✅ Done |
| 21 | D-21 | `scene_builder.py` | Far-bg vignette + front_fx CA | ✅ Done |
| 22 | D-22 | `image_io.py`, `scene_builder.py` | Display P3 color pipeline | ✅ Done |
| 23 | D-23 | `subject_runner.py` | BiRefNet-HR 1536×1536 | ✅ Done |
| 24 | D-24 | `depth_runner.py` | Depth Pro backend | ✅ Done |

**All 24 desktop issues complete.** Android-side enhancements tracked in `android/DESKTOP_HANDOFF.md`.

---

## Milestone 6: Visual Quality Enhancements

Targeted at producing **best-in-class output** for Samsung Galaxy S26 Ultra QHD+ AMOLED.
Every issue here is desktop-only — Android renderer enhancements are tracked in `android/DESKTOP_HANDOFF.md`.

### Issue D-19: LaMa Default Inpainter

**File:** `inpaint/inpainter.py`, `compose/scene_builder.py`

Currently `scene_builder.py` calls `inpaint_background(..., method="cv2")`.
OpenCV Telea produces visible smearing on large subject holes — artefacts become obvious during parallax shift (~80px on S26 Ultra).

**Change:** Switch default to `method="lama"` with automatic cv2 fallback if simple-lama-inpainting is not installed.
Move `simple-lama-inpainting` from optional `[inpaint]` group into core dependencies.

**Acceptance:** Inpainted background shows no visible smear artefacts on any of the 10 sample images.

---

### Issue D-20: Adaptive DOF Blur

**Files:** `compose/scene_builder.py`, `config.py`

Currently all background layers get a fixed `GaussianBlur(radius=2.5)`.
Real cameras produce depth-of-field where blur increases with distance from the focus plane.

**Change:** Compute per-layer blur from the actual depth band centre:
- `far_bg` → max blur (~4-6px, simulates distant defocus)
- `deep_mid` → moderate blur (~2-3px)
- `near_mid` → light blur (~0.5-1px)
- `hero_fg` → zero blur (sharp subject)
- `front_fx` → zero blur (edge fringe)

Add `max_blur_px: float = 6.0` to `PipelineConfig`. Compute `blur = band_depth * max_blur_px`.

**Acceptance:** Far background visually softer than near midground. Hero subject razor-sharp. Cinematic DOF feel.

---

### Issue D-21: Far-BG Vignette + Front-FX Chromatic Aberration

**File:** `compose/scene_builder.py`

Two AMOLED-targeted cinematic effects:

1. **Vignette on `layer_0_far_bg`:** Radial darkening towards corners (strength ~0.3). Pure blacks save AMOLED power and draw the eye inward.
2. **Chromatic aberration on `layer_4_front_fx`:** 1-2px RGB channel offset on the edge fringe. Gives a cinematic lens feel at near-zero perf cost.

Both are applied during `_render_assets()` before WebP encoding.

**Acceptance:** Far-bg corners noticeably darker. Front-fx edge fringe shows subtle RGB split visible at 100% zoom.

---

### Issue D-22: Display P3 Color Pipeline

**Files:** `utils/image_io.py`, `compose/scene_builder.py`

S26 Ultra has a P3 wide-gamut display. Current pipeline is sRGB throughout — reds and greens are muted compared to what the panel can display.

**Change:**
- `load_image_canvas()` → ICC-aware load via `PIL.ImageCms`. Preserve source profile, convert to linear sRGB for blending.
- All alpha compositing in linear light (not gamma-encoded).
- Final output → convert to Display P3 profile before WebP encode.
- Embed ICC profile in output WebP via Pillow's `icc_profile` parameter.

**Acceptance:** Output wallpapers render with visibly more vivid reds/greens on a P3-capable display vs the sRGB pipeline.

---

### Issue D-23: BiRefNet-HR 1536×1536

**File:** `segment/subject_runner.py`

Current BiRefNet processes at 1024×1024 input resolution. Higher-res input produces sharper edges on hair, fur, and thin structures.

**Change:** Add `birefnet_hr` model name that uses 1536×1536 input transforms while loading the same `ZhengPeng7/BiRefNet` weights. The model architecture is resolution-agnostic — only the preprocessing resize changes.

**Acceptance:** Hair/fur edges visually sharper on portrait images compared to 1024 variant. VRAM usage stays under 8GB.

---

### Issue D-24: Depth Pro Model Backend

**File:** `depth/depth_runner.py`

Apple's Depth Pro (open-sourced late 2024) produces **metric depth** with significantly sharper boundaries than Depth Anything V2, especially at object edges.

**Change:** Add `depth_pro` backend to `DepthRunner`. Pip: `depth-pro` or load from HuggingFace `apple/DepthPro`. Uses the same lazy-load pattern as existing backends.

**Acceptance:** Edge boundaries in depth map visibly crisper. Layer separation improved on scenes with foreground objects against sky.

---

---

## Android Side (Summary)

The Android app consumes the `corpus/` output or `.parallax` archive. It does NOT need any changes to handle real ML output vs stubs — the package format is the same. The Android implementation plan is tracked separately but depends on:

1. **CorpusLoader.kt** — unzip `.parallax`, parse `index.json` + per-wallpaper `meta.json`
2. **TextureLoader.kt** — decode 5 layer WebPs + clock_occlusion_mask into OpenGL textures
3. **SensorHandler.kt** — read TYPE_ROTATION_VECTOR, low-pass filter
4. **MotionController.kt** — critically damped spring interpolation per-layer
5. **GLRenderer.kt** — render 5 layers + clock plane in order with per-layer translation + alpha blend
6. **ClockRenderer.kt** — render time/date to texture, insert at `clock_plane_index`, apply occlusion mask
7. **ParallaxWallpaperService.kt** — `WallpaperService` implementation wiring GLRenderer + sensors
8. **UI** — picker (browse corpus), preview (static render), tuning (strength/speed sliders)

The desktop pipeline MUST work end-to-end first before Android rendering can be validated.
