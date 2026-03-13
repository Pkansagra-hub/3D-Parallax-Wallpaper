"""End-to-end integration test: full enhanced pipeline on all corpus images.

Uses heavy models:
  - Depth Anything V2 Large (depth)
  - BiRefNet (segmentation) with landscape detection
  - LaMa inpainting (auto-detect)
  - Adaptive DOF blur, vignette, chromatic aberration
  - Linear-light compositing, ICC profile embedding
"""

import logging
import time
from pathlib import Path

from parallaxgen.compose.scene_builder import build_scene_package
from parallaxgen.config import PipelineConfig
from parallaxgen.corpus.manifest import build_manifest, write_package_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

IMAGE_DIR = Path(__file__).parent / "src" / "parallaxgen" / "Images"
images = sorted(IMAGE_DIR.glob("*.png")) + sorted(IMAGE_DIR.glob("*.jpg"))
if not images:
    raise FileNotFoundError(f"No images in {IMAGE_DIR}")

print(f"\n{'='*70}")
print("  ParallaxGen — Full Enhanced Pipeline")
print(f"  Images: {len(images)}  |  Models: DAv2-Large + BiRefNet + LaMa")
print(f"{'='*70}\n")

config = PipelineConfig(
    depth_model="depth_anything_v2_large",
    segmentation_model="birefnet",
)

packages = []
total_t0 = time.perf_counter()

for idx, img_path in enumerate(images, 1):
    print(f"\n[{idx}/{len(images)}] {img_path.name}")
    print("-" * 60)

    t0 = time.perf_counter()
    try:
        package = build_scene_package(img_path, config=config)
    except Exception as e:
        print(f"  *** FAILED: {e}")
        continue
    elapsed = time.perf_counter() - t0

    packages.append(package)
    q = package.meta.quality
    landscape = (
        "LANDSCAPE" if package.meta.subject_bbox == (0.0, 0.0, 1.0, 1.0) else "SUBJECT"
    )

    print(f"  Mode         : {landscape}")
    print(f"  Time         : {elapsed:.2f}s")
    print(f"  Resolution   : {package.meta.resolution}")
    print(f"  Layers       : {len(package.layers)}")
    print(
        f"  Clock rect   : {tuple(round(v, 3) for v in package.meta.safe_clock_rect)}"
    )
    print(f"  Subject bbox : {tuple(round(v, 3) for v in package.meta.subject_bbox)}")
    print(f"  Quality pass : {q.get('passed', '?')}")
    print(f"  Depth sep    : {q.get('depth_separation', '?')}")
    print(f"  Subject cov  : {q.get('subject_coverage', '?')}")
    print(f"  Clock clear  : {q.get('clock_clearance', '?')}")
    if q.get("warnings"):
        for w in q["warnings"]:
            print(f"  ⚠ {w}")
    for name, data in sorted(package.rendered_assets.items()):
        print(f"    {name}: {len(data):,} bytes")

total_elapsed = time.perf_counter() - total_t0

# Write full corpus output
out_dir = Path(__file__).parent / "test_corpus"
for pkg in packages:
    write_package_files(out_dir, pkg)
manifest = build_manifest(packages)
manifest.write(out_dir / "index.json")

print(f"\n{'='*70}")
print("  BATCH COMPLETE")
print(f"  Images processed : {len(packages)}/{len(images)}")
print(f"  Total time       : {total_elapsed:.2f}s")
print(f"  Avg per image    : {total_elapsed / max(len(packages), 1):.2f}s")
print(f"  Corpus output    : {out_dir}")
print(f"{'='*70}")

# Summary table
print(f"\n{'Image':<55} {'Mode':<10} {'Pass':<6} {'Time':>6}")
print("-" * 80)
for pkg in packages:
    name = pkg.wallpaper_id[:52]
    q = pkg.meta.quality
    mode = "LAND" if pkg.meta.subject_bbox == (0.0, 0.0, 1.0, 1.0) else "SUBJ"
    passed = "✓" if q.get("passed") else "✗"
    print(f"  {name:<53} {mode:<10} {passed:<6}")
