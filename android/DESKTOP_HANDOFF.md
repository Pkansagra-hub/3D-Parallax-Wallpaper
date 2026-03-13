# Android Renderer — Enhancements from Desktop Review

These are **Android-side enhancements** identified during the desktop pipeline code review.
They are NOT part of the desktop Milestone 6 — they require Kotlin/OpenGL implementation
in the Android renderer.

The desktop pipeline ships corpus packages via `index.json` + per-wallpaper `meta.json`.
All rendering behaviour below should be driven by metadata already present in `meta.json`.

---

## 1. Spring Physics Parallax

**Current:** Direct gyro → layer offset mapping (linear).
**Target:** Critically damped spring interpolation for organic feel.

```
stiffness  ≈ 200
damping    ≈ 20
```

Implement in `MotionController.kt`. Each layer gets its own spring instance
scaled by `depth_weights[i]` from meta.json. The clock plane uses `clock_weight`.

---

## 2. 2D Vertical + Horizontal Parallax

**Current:** Horizontal-only parallax from gyro X axis.
**Target:** Full 2D parallax using both gyro X and Y.

Read `TYPE_ROTATION_VECTOR` for both axes in `SensorHandler.kt`.
Apply per-layer offset in both X and Y in `GLRenderer.kt`.
Vertical strength can be ~60% of horizontal to feel natural.

---

## 3. Non-Linear Depth Weight Curves

**Current:** Linear interpolation of `depth_weights` for layer offsets.
**Target:** Ease curve (ease-out cubic or similar) so:

- `layer_0_far_bg` moves at 1× (slow background drift)
- `layer_1_deep_mid` moves at ~1.5×
- `layer_2_near_mid` moves at ~2.5×
- `layer_3_hero_fg` anchored at 0× (stationary hero)
- `layer_4_front_fx` slight counter-movement (~-0.3×)

The weights in `meta.json` are the raw values. The ease curve is applied
by the renderer, not baked into the package.

---

## 4. HDR10+ Rendering

**Current:** SDR rendering pipeline.
**Target:** If device supports HDR10+ (S26 Ultra does), use `HardwareRenderer`
HDR path for wallpaper surface.

Desktop Milestone 6 may embed HDR gain maps in output WebP. The Android renderer
should detect and use them via `Gainmap` API (Android 14+).

---

## 5. Display P3 Color Rendering

Desktop Milestone 6 (D-22) outputs WebP with embedded Display P3 ICC profiles.
The Android renderer should:

- Detect P3 ICC profile in decoded Bitmap
- Use `ColorSpace.Named.DISPLAY_P3` when creating textures
- Ensure GL surface is configured for wide-gamut if available

---

## Priority Order

1. Spring physics (immediate feel improvement)
2. 2D parallax (dramatic quality jump)
3. Non-linear weight curves (subtle but polished)
4. P3 color rendering (matches desktop P3 output)
5. HDR10+ (future — requires desktop HDR pipeline first)

---

## Dependencies on Desktop

| Android Feature | Desktop Dependency | Status |
|---|---|---|
| Spring physics | `depth_weights` in meta.json | ✅ Already shipped |
| 2D parallax | No new metadata needed | ✅ Ready |
| Non-linear curves | `depth_weights` in meta.json | ✅ Already shipped |
| P3 color | D-22 (Display P3 pipeline) | 🔴 Milestone 6 |
| HDR10+ | Future desktop HDR pipeline | 🔴 Not started |
