# ParallaxGen

Spatial Depth Wallpaper System

Desktop AI Pipeline -> Android Spatial Renderer

## 1. Project Overview

ParallaxGen is a two-part system that converts a single photo into a premium spatial wallpaper package for Android. Heavy ML processing runs on a laptop, while the Android app renders a depth-aware 5-layer scene with gyroscope parallax and an inserted live clock plane.

The target visual language is closer to Apple spatial wallpapers and recent Samsung depth-style lock screen concepts than to a traditional flat live wallpaper. The wallpaper should feel cinematic, restrained, and composition-aware rather than exaggerated.

Core idea:

- Desktop app extracts depth, subject masks, occlusion masks, and composition metadata.
- Android app renders a precomputed spatial scene at 60fps with zero on-device ML inference.
- The clock is treated as its own render layer between depth groups.

## 2. Product Goals

Primary goals:

- Convert ordinary photos into premium spatial wallpapers.
- Support a 5-layer scene with clean foreground occlusion.
- Render a live clock between layers for an Apple-like depth effect.
- Maintain 60fps on modern Android devices.
- Keep all expensive ML work off-device.

Secondary goals:

- Provide a preview mode for composition QA before export.
- Allow user tuning for clock placement, strength, and motion feel.
- Support reusable wallpaper packages via a single `.parallax` archive.

Non-goals for v1:

- Full real-time ML inference on Android.
- Generic replacement of Samsung or stock Android SystemUI.
- Perfect lock-screen integration across all OEMs.
- Arbitrary video depth rendering.

## 3. Platform Strategy

Android imposes a hard boundary: a standard `WallpaperService` cannot reliably draw between native SystemUI lock-screen elements on all devices. Because of that, ParallaxGen should support two Android presentation modes.

Mode A: Spatial wallpaper mode

- Uses `WallpaperService`.
- Renders the 5-layer parallax scene.
- Best for home screen and wallpaper playback.
- Does not assume control over the OEM clock.

Mode B: Spatial clock mode

- Uses an app-controlled full-screen renderer or launcher-style surface.
- Draws ParallaxGen's own live clock between layers.
- Best for achieving the full depth-clock effect.
- Gives full control over typography, placement, occlusion, and motion.

Recommendation:

- Build the desktop pipeline once.
- Reuse the same `.parallax` package for both modes.

## 4. Visual Target

The effect should feel premium because of composition, not because of extreme motion.

Visual principles:

- Clean foreground separation.
- Large elegant clock typography.
- Selective occlusion, not full clock obstruction.
- Mild atmospheric depth, not artificial blur spam.
- Slow, damped motion.
- Strong subject placement and safe clock region.

The best input images usually have:

- one clear hero subject or silhouette,
- readable negative space for the clock,
- visible depth separation between foreground, midground, and background,
- clean edges around mountains, people, trees, buildings, or objects.

## 5. System Architecture

### 5.1 Full Pipeline Flow

```text
USER PHOTO (.jpg / .png)
        ->
[ DESKTOP ] Depth model -> normalized depth map
        ->
[ DESKTOP ] Subject segmentation + alpha matting
        ->
[ DESKTOP ] Smart layer composer -> 5 authored scene layers
        ->
[ DESKTOP ] Inpainting + edge cleanup
        ->
[ DESKTOP ] Clock composition analyzer -> safe region + occlusion mask
        ->
[ DESKTOP ] Package builder -> .parallax archive
        ->
[ ANDROID ] Corpus loader -> textures + metadata
        ->
[ ANDROID ] Sensor pipeline -> damped motion state
        ->
[ ANDROID ] OpenGL ES renderer -> spatial scene
        ->
[ ANDROID ] Optional clock renderer -> inserted between layers
```

### 5.2 Scene Model

ParallaxGen does not use equal depth slicing as the primary abstraction. It uses a fixed authored scene model with five visual layers plus one clock plane.

Render stack back to front:

1. Layer 0: far background
2. Layer 1: deep midground
3. Layer 2: near midground
4. Clock plane
5. Layer 3: hero foreground
6. Layer 4: front accents / thin occluders

This stack gives the clock enough depth presence to sit in the scene while still being partially occluded by the main subject.

### 5.3 Why 5 Layers

Five layers are the quality-performance sweet spot:

- enough separation for convincing depth,
- enough room to insert a clock plane cleanly,
- simpler than arbitrary N-slice rendering,
- manageable package size,
- easier per-image authoring and tuning.

## 6. Package Format

Each wallpaper is packaged as one ZIP-based `.parallax` archive.

Archive layout:

```text
wallpaper_package/
  index.json
  wall_001/
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
```

### 6.1 meta.json schema

```json
{
  "id": "wall_001",
  "version": 2,
  "resolution": [1080, 2400],
  "layer_count": 5,
  "clock_plane_index": 3,
  "parallax_strength": 0.65,
  "overscan": 0.18,
  "motion_profile": "cinematic_slow",
  "depth_weights": [0.08, 0.18, 0.32, 0.48, 0.62],
  "clock_weight": 0.24,
  "blur_px": [1.2, 0.8, 0.3, 0.0, 0.0],
  "safe_clock_rect": [0.16, 0.07, 0.84, 0.30],
  "focus_anchor": [0.50, 0.36],
  "subject_bbox": [0.27, 0.14, 0.73, 0.87],
  "has_clock_occlusion": true,
  "inpainted": true,
  "depth_model": "midas_dpt_large",
  "segmentation_model": "birefnet_or_equivalent"
}
```

### 6.2 index.json schema

```json
{
  "format": "parallaxgen-corpus-v2",
  "wallpapers": [
    {
      "id": "wall_001",
      "title": "Mountain Lake",
      "preview": "wall_001/preview.webp"
    }
  ]
}
```

## 7. Desktop Pipeline

### 7.1 Responsibilities

The desktop pipeline is responsible for all expensive or quality-sensitive operations:

- depth estimation,
- subject extraction,
- edge refinement,
- smart layer construction,
- inpainting,
- clock-safe composition analysis,
- packaging.

### 7.2 Recommended Tech Stack

| Library / Tool | Purpose |
| --- | --- |
| torch + torchvision | model execution |
| timm | model backbones |
| opencv-python | image processing |
| numpy | depth and mask operations |
| Pillow | WebP export |
| click or typer | CLI |
| tqdm | progress bars |
| simple-lama-inpainting or equivalent | hole filling |
| segmentation / matting model | subject extraction |

Python version:

- 3.10+

GPU strategy:

- depth and segmentation use CUDA when available,
- inpainting may run on CPU for v1.

### 7.3 Desktop Module Structure

```text
parallaxgen/
  cli.py
  depth/
    depth_runner.py
    depth_utils.py
  segment/
    subject_runner.py
    matte_refiner.py
  compose/
    layer_planner.py
    occlusion_planner.py
    scene_builder.py
  inpaint/
    inpainter.py
  corpus/
    manifest.py
    packer.py
  preview/
    preview_renderer.py
  utils/
    image_io.py
    geometry.py
```

### 7.4 CLI Commands

Process a single photo:

```bash
python -m parallaxgen process photo.jpg --output ./corpus
```

Batch process a folder:

```bash
python -m parallaxgen batch ./photos --output ./corpus
```

Preview the clock composition before packaging:

```bash
python -m parallaxgen preview photo.jpg --clock --output ./preview
```

Pack a corpus folder:

```bash
python -m parallaxgen pack ./corpus --out wallpapers.parallax.zip
```

Inspect package metadata:

```bash
python -m parallaxgen inspect ./corpus/wall_001.parallax
```

### 7.5 Processing Stages

Stage 1: Depth inference

- produce a normalized depth map in `[0.0, 1.0]`
- preserve strong discontinuities

Stage 2: Subject extraction

- isolate the hero subject
- generate a clean alpha matte
- refine hairlines, edges, and semi-transparent boundaries where possible

Stage 3: Layer planning

- do not split with `np.linspace` by default
- detect subject, horizon, depth histogram breaks, and empty clock space
- allocate pixels into the 5 authored layers

Stage 4: Inpainting

- fill holes opened by subject extraction
- largest fill effort on background layers
- light cleanup on middle layers
- no inpainting for front alpha accents unless needed

Stage 5: Clock composition analysis

- find a safe clock zone
- generate a clock occlusion mask
- reject images with poor readability

Stage 6: Packaging

- export WebP layers with alpha
- write `meta.json`
- write preview asset
- zip to `.parallax`

## 8. Smart Layering Logic

Equal depth bands are too crude for premium output. The layer builder should use semantic and compositional rules.

Priority rules:

- Hero subject gets its own foreground layer.
- Thin high-contrast edge details can be promoted to front accents.
- Background sky, walls, water, or distant landscape become stable low-motion layers.
- Midground is split only where it materially improves depth perception.

Practical heuristics:

- preserve major contours,
- penalize noisy tiny cutouts,
- favor large readable silhouettes,
- avoid placing heavy occlusion over the center of the clock,
- keep motion subtle for distant layers.

## 9. Android App

### 9.1 Responsibilities

The Android app is a high-performance renderer and package manager. It does not perform ML inference.

Responsibilities:

- load `.parallax` packages,
- decode WebP assets,
- update motion state from sensors,
- render layers in OpenGL ES 2.0,
- optionally render ParallaxGen's live clock,
- expose picker, preview, and tuning UI.

### 9.2 Android Tech Stack

| Component | Technology | Notes |
| --- | --- | --- |
| Language | Kotlin | Min SDK 26+ |
| Renderer | OpenGL ES 2.0 | broad compatibility |
| Sensors | TYPE_ROTATION_VECTOR | fused orientation data |
| UI | Jetpack Compose | picker and controls |
| Package loading | ZipInputStream | parse `.parallax` |
| Image decoding | BitmapFactory / ImageDecoder | asset loading |
| Clock rendering | Canvas to texture or glyph atlas | inserted render plane |

### 9.3 Android Module Structure

```text
app/src/main/
  corpus/
    CorpusLoader.kt
    CorpusManager.kt
    WallpaperMeta.kt
  motion/
    SensorHandler.kt
    MotionController.kt
    DampedSpring.kt
  renderer/
    GLRenderer.kt
    LayerShader.kt
    TextureLoader.kt
    SceneComposer.kt
    ClockRenderer.kt
  wallpaper/
    ParallaxWallpaperService.kt
    ParallaxWallpaperEngine.kt
  spatial/
    SpatialSceneActivity.kt
    SpatialSceneViewModel.kt
  ui/
    WallpaperPickerActivity.kt
    WallpaperPreviewScreen.kt
    TuningScreen.kt
```

## 10. Render Model

### 10.1 Motion Philosophy

Motion should feel slow, damped, and cinematic. Avoid arcade-like layer sliding.

### 10.2 Sensor Processing

Inputs:

- TYPE_ROTATION_VECTOR

Pipeline:

- raw orientation -> normalized tilt
- low-noise filtering
- critically damped interpolation
- per-layer weighted transform

### 10.3 Layer Weights

Suggested default weights:

- Layer 0 far background: 0.08
- Layer 1 deep midground: 0.18
- Layer 2 near midground: 0.32
- Clock plane: 0.24
- Layer 3 hero foreground: 0.48
- Layer 4 front accents: 0.62

The clock weight sits behind the foreground but ahead of the calmest layers so it feels embedded in the scene.

### 10.4 Transform Model

Each layer can apply:

- translation,
- tiny scale compensation,
- optional micro blur by depth,
- alpha blending.

Use overscan so edges never reveal:

- default overscan 18%

### 10.5 Render Order

```text
glClear(GL_COLOR_BUFFER_BIT)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

draw(layer0_farBackground)
draw(layer1_deepMidground)
draw(layer2_nearMidground)
draw(clockPlane)
draw(layer3_heroForeground)
draw(layer4_frontAccents)
```

## 11. Clock System

The clock is not a UI overlay added after the scene. It is a render plane inside the scene.

Requirements:

- live time and date updates,
- configurable typography,
- user-selectable placement inside `safe_clock_rect`,
- partial occlusion by hero subject,
- readable over varied backgrounds,
- low GPU overhead.

Clock styling controls:

- font family,
- weight,
- size,
- line height,
- opacity,
- glow or shadow,
- 12h / 24h format,
- date visibility.

Clock readability strategy:

- use `safe_clock_rect` for default placement,
- add mild shadow or glow if background contrast is weak,
- reduce overlap when occlusion exceeds threshold,
- allow per-wallpaper saved placement.

## 12. Quality Rules

Reject or warn on images that fail these checks:

- weak depth separation,
- messy subject edges,
- no usable safe clock space,
- excessive fine hair or foliage that creates noisy masks,
- severe inpainting artifacts,
- foreground covering too much of the clock zone.

Visual QA output should include:

- raw depth preview,
- subject mask preview,
- layer stack preview,
- clock occlusion preview,
- animated parallax preview.

## 13. Recommended Build Order

Phase 1: Core desktop pipeline

- set up Python project and CLI skeleton
- integrate depth model
- integrate subject segmentation and matte refinement
- generate 5 scene layers
- add inpainting
- export `.parallax` package

Phase 2: Static Android renderer

- parse package files
- load textures
- render static 5-layer scene in OpenGL ES 2.0
- validate layer order and alpha quality

Phase 3: Motion system

- add sensor input
- implement damped motion controller
- apply per-layer transforms
- tune overscan and default weights

Phase 4: Clock insertion

- build clock renderer
- place clock between layers
- apply occlusion mask
- tune readability and style presets

Phase 5: Product polish

- picker UI
- preview screen
- tuning controls
- package import flow
- performance profiling and battery testing

## 14. Key Design Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Scene structure | Fixed 5 layers + clock plane | predictable quality and manageable runtime |
| ML location | Desktop only | best quality without mobile ML cost |
| Render API | OpenGL ES 2.0 | wide device support |
| Image format | WebP with alpha | compact and Android-friendly |
| Clock model | In-scene render plane | required for true occlusion effect |
| Motion style | Damped and restrained | premium look over exaggerated movement |
| Layer planning | semantic + compositional | better than equal depth bands |

## 15. Risks

Main technical risks:

- poor subject matting on difficult images,
- visible inpainting artifacts,
- OEM-specific wallpaper behavior,
- clock readability on busy scenes,
- package size growth with five alpha layers.

Mitigations:

- reject low-quality inputs,
- add manual tuning hooks later,
- keep a renderer mode that does not depend on OEM lock-screen behavior,
- use adaptive compression and preview scoring.

## 16. Success Criteria

ParallaxGen v2 is successful if it can:

- turn a good input photo into a convincing spatial package in one command,
- render 5 layers smoothly at 60fps,
- place a live clock between layers with believable occlusion,
- look premium on both still preview and motion,
- outperform sticker-based or fake depth workarounds in quality and realism.

## 17. Future Extensions

- desktop depth editor for manual mask correction
- composition assistant that recommends best clock zone
- Wi-Fi sync from desktop to phone
- launcher mode with minimal icons and immersive UI
- animated weather or particle overlays
- video-derived spatial scenes
- OEM-specific integrations where available

ParallaxGen - Project Spec v2.0
