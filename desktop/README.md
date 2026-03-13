# ParallaxGen — Desktop Pipeline

Generate premium **spatial depth wallpaper packages** from single photos for
the ParallaxGen Android renderer.  Each package ships 5 RGBA WebP layers, a
clock-occlusion mask, and per-wallpaper metadata that drives parallax motion
and dynamic clock rendering on-device.

Target device: **Samsung Galaxy S26 Ultra** (1440 × 3120).

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.13 |
| GPU | — (CPU works) | NVIDIA with CUDA 12+ |
| VRAM | — | ≥ 6 GB (Depth Anything V2 Large + BiRefNet) |
| RAM | 8 GB | 16 GB |
| Disk | ~4 GB (model cache) | SSD |

## Installation

```bash
# Clone the repo and enter the desktop directory
cd desktop

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# Install with pip (editable for development)
pip install -e .

# (Optional) ML inpainting with LaMa
pip install -e ".[inpaint]"
```

### CUDA Setup

Install PyTorch with CUDA **before** the package if you want GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Quick Start

### Process a single image

```bash
parallaxgen process photo.png --output corpus/
```

### Batch-process a directory

```bash
parallaxgen batch ./Images --output corpus/
```

### Benchmark throughput

```bash
parallaxgen benchmark ./Images
```

### Preview (inspection only — no full package)

```bash
parallaxgen preview photo.png --clock
```

### Pack corpus into `.parallax` archive

```bash
parallaxgen pack corpus/ --out wallpapers.parallax
```

### Inspect a generated meta.json

```bash
parallaxgen inspect corpus/desert_mountains/
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `process` | Generate one wallpaper package from a single image |
| `batch` | Process every image in a directory, write corpus + `index.json` |
| `benchmark` | Time each image, print summary table (output discarded) |
| `preview` | Quick JSON summary: depth model, segmentation, quality scores |
| `pack` | ZIP a prepared corpus directory into a `.parallax` archive |
| `inspect` | Print the `meta.json` for a given wallpaper directory |

### Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--width` | 1440 | Output width in pixels |
| `--height` | 3120 | Output height in pixels |
| `--depth-model` | `depth_anything_v2_large` | Depth estimation model |
| `--segmentation-model` | `birefnet_or_equivalent` | Subject segmentation model |
| `--overscan` | 0.18 | Extra canvas for parallax shift headroom |
| `--parallax-strength` | 0.65 | Overall motion intensity |

## Package Format (`.parallax` v2)

```
corpus/
├── index.json                      # Corpus manifest
├── wallpaper_id/
│   ├── meta.json                   # Per-wallpaper metadata
│   ├── layer_0_far_bg.webp         # Furthest background (blurred, inpainted)
│   ├── layer_1_deep_mid.webp       # Deep midground
│   ├── layer_2_near_mid.webp       # Near midground
│   ├── layer_3_hero_fg.webp        # Hero foreground (subject)
│   ├── layer_4_front_fx.webp       # Edge fringe / bokeh shell
│   ├── clock_occlusion_mask.webp   # Alpha mask for clock-behind-subject
│   ├── subject_mask.webp           # Subject alpha matte
│   ├── depth_map.webp              # Normalised depth (0=near 1=far)
│   ├── preview.webp                # Composite preview with clock rect
│   └── qa_grid.webp                # 2×2 diagnostic grid (dev only)
```

### meta.json Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique wallpaper identifier |
| `version` | int | Package format version (2) |
| `resolution` | [int, int] | Width × height |
| `layer_count` | int | Always 5 |
| `clock_plane_index` | int | Render stack position (3) |
| `clock_weight` | float | Parallax weight for the clock plane |
| `clock_font_scale` | float | Clock digit size relative to viewport |
| `clock_anchor` | [float, float] | Normalised (cx, cy) centre |
| `safe_clock_rect` | [float, float, float, float] | (l, t, r, b) safe zone |
| `has_clock_occlusion` | bool | Ship clock occlusion mask? |
| `parallax_strength` | float | Overall parallax intensity |
| `depth_weights` | [float × 5] | Per-layer parallax weights |
| `blur_px` | [float × 5] | Per-layer blur radii |
| `quality` | object | Quality scorer output |

## ML Models

| Model | Task | Params | Source |
|-------|------|--------|--------|
| Depth Anything V2 Large | Monocular depth | 335 M | `depth-anything/Depth-Anything-V2-Large-hf` |
| BiRefNet | Salient object segmentation | ~80 M | `ZhengPeng7/BiRefNet` |
| OpenCV Telea | Background inpainting | — | `cv2.inpaint` (built-in) |
| LaMa (optional) | ML inpainting | ~27 M | `simple-lama-inpainting` |

Models auto-download to the HuggingFace cache on first run (~2 GB total).

## Quality Scoring

Every processed wallpaper is scored on three axes:

1. **Depth separation** — dynamic range used in the depth map (p5–p95)
2. **Mask cleanliness** — perimeter-to-area ratio of the subject contour
3. **Clock readability** — subject occlusion inside the safe clock rect

A wallpaper **passes** if all three scores exceed their thresholds.  Warnings
are logged for any failing axis.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Use `--depth-model depth_anything_v2_base` or `_small` |
| Slow first run | Model download; subsequent runs use cache |
| `ModuleNotFoundError: kornia` | `pip install kornia>=0.7 timm>=1.0` |
| Noisy subject edges | Expected for complex scenes; matte refiner applies morphological cleanup |
| `Missing rendered assets` | Pipeline bug — check logs for errors in earlier stages |

## License

Private — not yet released.
