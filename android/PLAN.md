# Android Plan — End-to-End Implementation

This document is the single source of truth for every component in the ParallaxGen Android renderer. Every module has exact class names, dependencies, implementation details, and acceptance criteria. No stubs. No "choose later". Every decision is made here.

---

## Current State (Honest Assessment)

**Existing scaffolding — all 18 .kt files are stubs:**

- `corpus/` — CorpusLoader returns dummy `WallpaperMeta(id = root.name)`, no JSON parsing
- `motion/` — DampedSpring and MotionController have correct math but never wired to sensors
- `renderer/` — GLRenderer, SceneComposer, ClockRenderer are placeholder classes with no OpenGL
- `spatial/` — SpatialSceneActivity shows a `Text("Spatial scene preview surface")` placeholder
- `ui/` — WallpaperPickerActivity, WallpaperPreviewScreen, TuningScreen are text-only Compose stubs
- `wallpaper/` — ParallaxWallpaperService creates an engine that does nothing on `onVisibilityChanged`

**Gradle configured:** AGP 8.7.3, Kotlin 2.0.21, compileSdk 35, minSdk 26, Compose enabled, Material3 deps present.

**AndroidManifest.xml:** Has launcher activity, spatial activity, and WallpaperService with BIND_WALLPAPER permission. Missing `<meta-data>` for wallpaper XML.

**Desktop output ready:** 10+ wallpaper packages in `test_corpus/` with valid `meta.json`, 5 RGBA WebP layers, clock occlusion mask, depth map, subject mask, preview, QA grid, and `index.json`.

---

## What We're Building

A complete Android app that:

1. Loads `.parallax` corpus packages from bundled assets or device storage
2. Renders 5 RGBA layers + clock plane at 60fps via OpenGL ES 3.0
3. Applies gyroscope-driven parallax with spring physics
4. Draws a live clock between layers 2 and 3 with partial occlusion
5. Provides a picker UI to browse, preview, and set wallpapers
6. Runs as both a standalone preview activity AND a `WallpaperService`
7. **Runs in Android Emulator inside VS Code** before final APK build

---

## Testing Strategy — Run in VS Code

Before building the APK, we'll validate the app in the Android Emulator from VS Code:

1. Install the **Android** VS Code extension (by Google)
2. Create an AVD (Pixel 8 Pro, API 35, x86_64) via AVD Manager
3. Build with `./gradlew assembleDebug`
4. Install with `adb install app/build/outputs/apk/debug/app-debug.apk`
5. Launch picker activity via `adb shell am start -n com.parallaxgen/.ui.WallpaperPickerActivity`
6. Push test corpus via `adb push ../desktop/test_corpus/ /sdcard/ParallaxGen/corpus/`
7. Validate rendering, parallax motion (use emulator virtual sensors), and clock display
8. Set as wallpaper and verify WallpaperService works

Final APK: `./gradlew assembleRelease` → `app/build/outputs/apk/release/app-release-unsigned.apk`

---

## Milestone 1: Real Corpus Loading & Data Model (A-1 through A-3) ✅ COMPLETE

### Issue A-1: Parse real meta.json into WallpaperMeta

**Files:** `WallpaperMeta.kt`, `CorpusLoader.kt`

**What changes:**

- `WallpaperMeta` must parse the EXACT schema from desktop's `meta.json`:

  ```json
  {
    "version": 2,
    "target_device": "Galaxy S26 Ultra",
    "resolution": [1440, 3120],
    "layer_count": 5,
    "clock_plane_index": 3,
    "parallax_strength": 0.65,
    "overscan": 0.18,
    "motion_profile": "cinematic_slow",
    "depth_weights": [0.08, 0.18, 0.32, 0.48, 0.62],
    "blur_px": [1.2, 0.8, 0.3, 0.0, 0.0],
    "clock_weight": 0.24,
    "clock_font_scale": 0.62,
    "clock_anchor": [0.5, 0.22],
    "safe_clock_rect": [0.16, 0.07, 0.84, 0.30],
    "has_clock_occlusion": true,
    "focus_anchor": [0.5, 0.36],
    "subject_bbox": [0.0, 0.288, 0.999, 1.0],
    "inpainted": true,
    "depth_model": "depth_anything_v2_large",
    "segmentation_model": "birefnet",
    "quality": {
      "depth_separation": 0.62,
      "mask_cleanliness": 0.86,
      "clock_readability": 1.0,
      "warnings": [],
      "passed": true
    },
    "id": "desert-mountain-galaxy-wallpaper-by-one4wall-app"
  }
  ```

- Use `org.json.JSONObject` (built into Android SDK, no extra deps)
- `CorpusLoader.loadPackage(dir: File)` reads `meta.json`, decodes all fields, returns validated `WallpaperMeta`
- Fail gracefully: if `meta.json` missing or malformed, skip that wallpaper with a log warning

**Acceptance:** Load 10 desktop test_corpus wallpapers → all 10 return valid `WallpaperMeta` with correct field values.

---

### Issue A-2: Parse index.json and manage corpus directory

**Files:** `CorpusManager.kt`, `CorpusLoader.kt`

**What changes:**

- `CorpusLoader.loadIndex(corpusDir: File): List<CorpusEntry>` parses `index.json`
- `CorpusEntry` data class: `id`, `title`, `previewPath`
- `CorpusManager.loadCorpus(baseDir: File): List<WallpaperPackage>` joins index entries with loaded `WallpaperMeta`
- `WallpaperPackage` data class: `meta: WallpaperMeta`, `title: String`, `directory: File`, `previewFile: File`
- Support two corpus sources:
  1. Bundled in `assets/corpus/` (for built-in wallpapers)
  2. External storage at `/sdcard/ParallaxGen/corpus/` (for user-pushed packages)

**Acceptance:** `CorpusManager` lists all wallpapers from both sources, correct titles and preview paths.

---

### Issue A-3: Bundle test corpus into assets

**Files:** `app/src/main/assets/corpus/` directory

**What changes:**

- Copy 3 representative wallpaper packages from `desktop/test_corpus/` into `app/src/main/assets/corpus/`
  - Pick: desert-mountain-galaxy, ocean-lagoon, snowy-mountain (one SUBJECT, one LANDSCAPE, one complex)
- Include their `meta.json` + all WebP assets + the top-level `index.json` (trimmed to 3 entries)
- `CorpusLoader` adds `loadFromAssets(context: Context)` method using `AssetManager`

**Acceptance:** App launches with 3 wallpapers available without any ADB push.

---

## Milestone 2: OpenGL ES Renderer — Static Scene (A-4 through A-7) ✅ COMPLETE

### Issue A-4: EGL surface + GLSurfaceView setup

**Files:** `renderer/ParallaxGLSurfaceView.kt`, `renderer/ParallaxGLRenderer.kt`

**What changes:**

- New `ParallaxGLSurfaceView` extends `GLSurfaceView`, sets OpenGL ES 3.0, RGBA 8888
- `ParallaxGLRenderer` implements `GLSurfaceView.Renderer`
  - `onSurfaceCreated`: compile shaders, init GL state
  - `onSurfaceChanged`: set viewport, compute projection
  - `onDrawFrame`: render scene
- Wire into `SpatialSceneActivity` replacing the placeholder Text composable
- Use `AndroidView` in Compose to embed the GLSurfaceView

**Acceptance:** Black GL surface appears in SpatialSceneActivity, no crash.

---

### Issue A-5: Real GLSL shaders with texture + transform

**File:** `renderer/LayerShader.kt`

**What changes:**

- Replace stub shaders with real GLSL:

  ```glsl
  // Vertex
  attribute vec4 aPosition;
  attribute vec2 aTexCoord;
  uniform mat4 uMVPMatrix;
  uniform vec2 uOffset;
  uniform float uScale;
  varying vec2 vTexCoord;
  void main() {
      vec2 pos = aPosition.xy * uScale + uOffset;
      gl_Position = uMVPMatrix * vec4(pos, 0.0, 1.0);
      vTexCoord = aTexCoord;
  }

  // Fragment
  precision mediump float;
  uniform sampler2D uTexture;
  varying vec2 vTexCoord;
  void main() {
      gl_FragColor = texture2D(uTexture, vTexCoord);
  }
  ```

- Shader compilation utility with error logging
- Uniform locations cached after link

**Acceptance:** Shader compiles and links without errors on GL ES 3.0.

---

### Issue A-6: Real texture loading from WebP layers

**File:** `renderer/TextureLoader.kt`

**What changes:**

- `loadTexture(file: File): Int` — decode WebP via `BitmapFactory`, upload to GL texture
- `loadTextureFromAssets(am: AssetManager, path: String): Int` — same for bundled assets
- Handle RGBA (layers with alpha) and RGB (depth map, preview)
- Use `GLES30.glGenTextures`, `GLUtils.texImage2D`, set min/mag filter to `GL_LINEAR`, wrap to `GL_CLAMP_TO_EDGE`
- Recycle bitmap after upload to save memory
- `SceneTextures` data class holds all 5 layer texture IDs + clock occlusion mask texture ID

**Acceptance:** All 5 layers + occlusion mask loaded as GL textures, non-zero IDs.

---

### Issue A-7: Static 5-layer render with correct draw order

**Files:** `renderer/ParallaxGLRenderer.kt`, `renderer/SceneComposer.kt`

**What changes:**

- `SceneComposer.loadScene(meta: WallpaperMeta, textures: SceneTextures)` builds render list
- `ParallaxGLRenderer.onDrawFrame()`:
  1. `glClear(GL_COLOR_BUFFER_BIT)`
  2. `glEnable(GL_BLEND)` + `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)`
  3. Draw layers 0-4 back-to-front with overscan quad geometry
  4. Each layer is a fullscreen textured quad with 18% overscan padding
- All layers rendered at identity transform (no motion yet — static proof)

**Acceptance:** SpatialSceneActivity shows the wallpaper with all 5 layers composited. Visually matches desktop's `preview.webp`.

---

## Milestone 3: Motion System — Gyroscope Parallax (A-8 through A-10) ✅ COMPLETE

### Issue A-8: Real sensor registration with TYPE_ROTATION_VECTOR

**File:** `motion/SensorHandler.kt`

**What changes:**

- Register `TYPE_ROTATION_VECTOR` sensor in `onResume`, unregister in `onPause`
- Extract pitch and roll from rotation matrix
- Normalize to [-1, 1] range with configurable sensitivity
- Provide callback: `onTiltChanged(tiltX: Float, tiltY: Float)`
- Handle missing gyroscope gracefully (emulator fallback: use `TYPE_ACCELEROMETER`)

**Acceptance:** Tilting device/emulator produces smooth tilt values in [-1, 1].

---

### Issue A-9: Spring-physics motion controller wired to renderer

**Files:** `motion/MotionController.kt`, `motion/DampedSpring.kt`

**What changes:**

- `DampedSpring` already has correct math — keep as-is
- `MotionController` gets per-layer spring instances:
  - One `DampedSpring` pair (X, Y) per layer, stiffness/damping from meta
  - `update()` produces `LayerMotionState` array: per-layer (offsetX, offsetY)
  - Layer offsets scaled by `depth_weights[i]` from meta.json
  - Clock plane uses `clock_weight`
  - Non-linear ease curve: apply ease-out cubic on raw weights
- Vertical parallax at 60% of horizontal strength

**Acceptance:** Tilting emulator produces smooth per-layer offsets with correct weight scaling.

---

### Issue A-10: Apply motion offsets in GL renderer

**File:** `renderer/ParallaxGLRenderer.kt`

**What changes:**

- Each layer's `uOffset` uniform set from `LayerMotionState[i]`
- `uScale` = `1.0 + overscan` (default 1.18) to prevent edge reveal
- Per-frame update loop: sensor → motion controller → per-layer offsets → GL uniforms
- Hero foreground (layer 3) stationary: offset = 0
- Front accents (layer 4) subtle counter-movement: offset = -0.3× base

**Acceptance:** All 5 layers move at different speeds when tilting. Far background drifts slow, near layers move more. Hero stays anchored. Clock plane sits between.

---

## Milestone 4: Clock Rendering System (A-11 through A-13)

### Issue A-11: Clock texture from Canvas

**File:** `renderer/ClockRenderer.kt`

**What changes:**

- Create `Bitmap` + `Canvas` at wallpaper resolution
- Draw time (HH:mm) with large thin font — `Typeface.create("sans-serif-light", Typeface.NORMAL)`
- Draw date (EEE, MMM d) below at smaller size
- Font scale from `meta.clock_font_scale`
- Position from `meta.clock_anchor` (normalized → pixel coords)
- White text with subtle drop shadow for readability
- Upload result to GL texture each minute (or on time change)
- `ClockSnapshot` replaced with `clockTextureId: Int`

**Acceptance:** Clock displays correct time, updates every minute, positioned within safe_clock_rect.

---

### Issue A-12: Clock occlusion via mask

**Files:** `renderer/ParallaxGLRenderer.kt`

**What changes:**

- Load `clock_occlusion_mask.webp` as GL texture
- When drawing clock plane:
  1. Draw clock texture
  2. Apply occlusion mask as alpha multiply (where mask is white → clock hidden)
  3. This makes the subject partially cover the clock
- Fragment shader variant for clock:

  ```glsl
  uniform sampler2D uClockTexture;
  uniform sampler2D uOcclusionMask;
  varying vec2 vTexCoord;
  void main() {
      vec4 clock = texture2D(uClockTexture, vTexCoord);
      float mask = texture2D(uOcclusionMask, vTexCoord).r;
      gl_FragColor = vec4(clock.rgb, clock.a * (1.0 - mask));
  }
  ```

**Acceptance:** Clock is partially hidden behind the hero subject. Matches desktop QA grid occlusion preview.

---

### Issue A-13: Clock in render pipeline between layers 2 and 3

**File:** `renderer/ParallaxGLRenderer.kt`

**What changes:**

- Draw order becomes:
  1. Layer 0 (far bg)
  2. Layer 1 (deep mid)
  3. Layer 2 (near mid)
  4. **Clock plane** (with occlusion mask)
  5. Layer 3 (hero fg)
  6. Layer 4 (front fx)
- Clock plane gets its own parallax offset from `clock_weight`
- Clock auto-updates: check `System.currentTimeMillis()` each frame, regenerate texture when minute changes

**Acceptance:** Clock renders between the correct layers, moves with scene parallax, partially occluded by hero. Time is live and accurate.

---

## Milestone 5: WallpaperService Integration (A-14 through A-15)

### Issue A-14: Real WallpaperService with EGL context

**Files:** `wallpaper/ParallaxWallpaperService.kt`, `wallpaper/ParallaxWallpaperEngine.kt`

**What changes:**

- `EngineImpl` extends `Engine()` and manages its own EGL context (no GLSurfaceView available in WallpaperService)
- Manual EGL setup: `eglGetDisplay → eglInitialize → eglChooseConfig → eglCreateContext → eglCreateWindowSurface`
- Render thread with `Choreographer.FrameCallback` for 60fps vsync
- Share renderer code with `ParallaxGLRenderer` (same draw calls)
- Load wallpaper from `CorpusManager` based on user preference (SharedPreferences)
- `onVisibilityChanged(true)` → start render thread
- `onVisibilityChanged(false)` → pause (stop posting frame callbacks)
- `onDestroy` → tear down EGL + release textures

**Acceptance:** Set wallpaper via system picker → parallax scene renders on home screen at 60fps.

---

### Issue A-15: Wallpaper metadata XML + system integration

**Files:** `res/xml/wallpaper.xml`, `AndroidManifest.xml`

**What changes:**

- Create `res/xml/wallpaper.xml`:

  ```xml
  <?xml version="1.0" encoding="utf-8"?>
  <wallpaper xmlns:android="http://schemas.android.com/apk/res/android"
      android:thumbnail="@drawable/wallpaper_thumb"
      android:description="@string/wallpaper_description" />
  ```

- Add `<meta-data>` to service in manifest:

  ```xml
  <meta-data android:name="android.service.wallpaper"
             android:resource="@xml/wallpaper" />
  ```

- Add `@drawable/wallpaper_thumb` (use one of the preview.webp images)
- Add string resource for description

**Acceptance:** ParallaxGen appears in system wallpaper picker under "Live Wallpapers".

---

## Milestone 6: Picker UI & Preview (A-16 through A-18)

### Issue A-16: Wallpaper grid picker with preview thumbnails

**File:** `ui/WallpaperPickerActivity.kt`, `ui/WallpaperPreviewScreen.kt`

**What changes:**

- `WallpaperPickerActivity`:
  - Load all wallpapers from `CorpusManager`
  - Display grid of preview thumbnails (LazyVerticalGrid, 2 columns)
  - Each card shows `preview.webp` with title overlay
  - Tap → navigate to preview screen
- `WallpaperPreviewScreen`:
  - Full-screen GLSurfaceView showing the selected wallpaper with live parallax
  - Overlay buttons: "Set as Wallpaper", "Back"
  - "Set as Wallpaper" saves selection to SharedPreferences + calls `WallpaperManager.setWallpaper`

**Acceptance:** User sees grid of wallpapers, taps one, sees full live preview with parallax, can set as wallpaper.

---

### Issue A-17: Import wallpapers from storage

**File:** `ui/WallpaperPickerActivity.kt`, `corpus/CorpusManager.kt`

**What changes:**

- "Import" FAB button in picker
- Uses `ActivityResultContracts.OpenDocumentTree` to pick a corpus directory
- OR: `OpenDocument` to pick a `.parallax.zip` file
- Unzip/copy to app's internal storage `filesDir/corpus/`
- Refresh wallpaper grid after import

**Acceptance:** User can ADB-push or file-picker import new wallpaper packages and they appear in the grid.

---

### Issue A-18: Tuning controls

**File:** `ui/TuningScreen.kt`

**What changes:**

- Accessible from preview screen via gear icon
- Sliders for:
  - Parallax strength (0.0–1.0, default from meta)
  - Clock opacity (0.0–1.0, default 0.9)
  - Motion damping (low/medium/high → spring stiffness presets)
- Toggle: 12h / 24h clock format
- Toggle: Show/hide date line
- Save to SharedPreferences per-wallpaper

**Acceptance:** Changing sliders live-updates the preview. Settings persist across app restarts.

---

## Milestone 7: Polish, Emulator Validation & APK Build (A-19 through A-22)

### Issue A-19: Performance profiling + battery optimization

**What changes:**

- Profile with `systrace` / GPU profiler in Android Studio or via `adb shell dumpsys gfxinfo`
- Target: <16ms frame time (60fps) on Pixel 8 Pro emulator
- Reduce texture memory: downscale layers to screen resolution if source is larger
- Skip rendering when wallpaper not visible (`onVisibilityChanged(false)`)
- Release textures when switching wallpapers

**Acceptance:** Consistent 60fps in emulator, no jank frames in 30-second capture.

---

### Issue A-20: Edge cases + error handling

**What changes:**

- No gyroscope: fall back to accelerometer, or disable parallax with static scene
- Corrupt/missing meta.json: skip wallpaper, show toast
- OOM on texture load: downscale to 50% and retry once
- Surface lost during wallpaper switch: recreate EGL context
- Empty corpus: show "Import wallpapers" prompt instead of empty grid

**Acceptance:** App doesn't crash on any of the above scenarios.

---

### Issue A-21: Run & validate in VS Code Android Emulator

**What changes:**

- Verify Android SDK + emulator installed
- Create AVD: Pixel 8 Pro, API 35, system image `google_apis;x86_64`
- Build: `cd android && ./gradlew assembleDebug`
- Install: `adb install -r app/build/outputs/apk/debug/app-debug.apk`
- Push corpus: `adb push ../desktop/test_corpus/ /sdcard/ParallaxGen/corpus/`
- Launch: `adb shell am start -n com.parallaxgen/.ui.WallpaperPickerActivity`
- Validate:
  1. Picker shows wallpaper grid with previews ✓
  2. Tap wallpaper → live GL preview with parallax ✓
  3. Clock displays between layers with occlusion ✓
  4. "Set as Wallpaper" works ✓
  5. Home screen shows parallax wallpaper ✓
  6. Virtual sensors produce motion ✓

**Acceptance:** Full user flow works in emulator. Screenshots captured for reference.

---

### Issue A-22: Build final release APK

**What changes:**

- `app/build.gradle.kts`:
  - Enable `isMinifyEnabled = true` for release
  - Add R8 ProGuard rules for OpenGL, JSON, Compose
- Build: `./gradlew assembleRelease`
- Output: `app/build/outputs/apk/release/app-release-unsigned.apk`
- Optional: sign with debug key for sideloading:

  ```bash
  apksigner sign --ks ~/.android/debug.keystore --ks-pass pass:android app-release.apk
  ```

- Final artifact size target: <20MB (3 bundled wallpapers + app code)

**Acceptance:** APK installs and runs correctly on emulator. Same behavior as debug build.

---

## Implementation Order

```
Milestone 1: Corpus Loading        [A-1, A-2, A-3]     — data model + real parsing
Milestone 2: GL Renderer           [A-4, A-5, A-6, A-7] — see layers on screen
Milestone 3: Motion System         [A-8, A-9, A-10]     — parallax from sensors
Milestone 4: Clock System          [A-11, A-12, A-13]   — live clock with occlusion
Milestone 5: WallpaperService      [A-14, A-15]         — system wallpaper integration
Milestone 6: Picker UI             [A-16, A-17, A-18]   — browse, preview, import, tune
Milestone 7: Polish + APK          [A-19, A-20, A-21, A-22] — perf, errors, emulator test, release
```

Total: 22 issues across 7 milestones.

---

## Dependencies

Already in `app/build.gradle.kts`:

- `androidx.activity:activity-compose`
- `androidx.compose.material3:material3`
- `androidx.compose.ui:ui`
- `androidx.core:core-ktx`
- `androidx.lifecycle:lifecycle-runtime-ktx`
- `androidx.lifecycle:lifecycle-viewmodel-compose`

**Need to add:**

- `androidx.compose.foundation:foundation` (LazyVerticalGrid)
- `io.coil-kt:coil-compose:2.6.0` (image loading for thumbnails in picker)
- No additional deps for OpenGL ES (part of Android SDK)
- No additional deps for JSON (org.json is in Android SDK)
- No additional deps for sensors (android.hardware.Sensor is in SDK)

---

## File Inventory — New & Modified

| File | Action | Milestone |
|---|---|---|
| `corpus/WallpaperMeta.kt` | Rewrite — real JSON parsing | M1 |
| `corpus/CorpusLoader.kt` | Rewrite — parse meta.json + index.json | M1 |
| `corpus/CorpusManager.kt` | Rewrite — dual source (assets + storage) | M1 |
| `corpus/CorpusEntry.kt` | **New** — index.json entry model | M1 |
| `corpus/WallpaperPackage.kt` | **New** — full package model | M1 |
| `assets/corpus/` | **New** — 3 bundled wallpapers | M1 |
| `renderer/ParallaxGLSurfaceView.kt` | **New** — GL surface setup | M2 |
| `renderer/ParallaxGLRenderer.kt` | **New** — real GL renderer | M2 |
| `renderer/LayerShader.kt` | Rewrite — real GLSL shaders | M2 |
| `renderer/TextureLoader.kt` | Rewrite — real texture upload | M2 |
| `renderer/SceneComposer.kt` | Rewrite — real scene composition | M2 |
| `renderer/SceneTextures.kt` | **New** — texture ID holder | M2 |
| `motion/SensorHandler.kt` | Rewrite — real sensor registration | M3 |
| `motion/MotionController.kt` | Rewrite — per-layer spring offsets | M3 |
| `renderer/ClockRenderer.kt` | Rewrite — Canvas→texture clock | M4 |
| `wallpaper/ParallaxWallpaperService.kt` | Rewrite — real EGL + render thread | M5 |
| `wallpaper/ParallaxWallpaperEngine.kt` | Rewrite — real engine loop | M5 |
| `res/xml/wallpaper.xml` | **New** — wallpaper metadata | M5 |
| `ui/WallpaperPickerActivity.kt` | Rewrite — grid picker UI | M6 |
| `ui/WallpaperPreviewScreen.kt` | Rewrite — live GL preview | M6 |
| `ui/TuningScreen.kt` | Rewrite — real slider controls | M6 |
| `spatial/SpatialSceneActivity.kt` | Rewrite — embed GLSurfaceView | M2 |
| `spatial/SpatialSceneViewModel.kt` | Rewrite — hold loaded scene state | M2 |
| `AndroidManifest.xml` | Update — wallpaper meta-data, storage permissions | M5 |
| `app/build.gradle.kts` | Update — add deps, R8 config | M1/M7 |

---

## Acceptance Criteria — Final

The Android app is done when:

1. **Picker** shows grid of wallpaper previews (bundled + imported)
2. **Preview** renders live 5-layer parallax scene at 60fps
3. **Clock** displays between layers 2 and 3, partially occluded by hero subject
4. **Motion** responds to device tilt with spring-damped parallax
5. **WallpaperService** works as system live wallpaper
6. **Import** allows adding new `.parallax` packages
7. **Tuning** controls adjust parallax strength, clock style, motion feel
8. **Emulator** validation passes all checks in A-21
9. **APK** builds and installs successfully
