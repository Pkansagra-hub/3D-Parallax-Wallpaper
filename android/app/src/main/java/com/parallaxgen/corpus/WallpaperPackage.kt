package com.parallaxgen.corpus

import java.io.File

/** A fully loaded wallpaper package with metadata and file references. */
data class WallpaperPackage(
    val meta: WallpaperMeta,
    val title: String,
    val directory: File?,
    val previewFile: File?,
    val isAsset: Boolean = false,
    val assetBasePath: String? = null,
) {
    /** Layer asset filenames in render order (0–4). */
    val layerFiles: List<String>
        get() = listOf(
            "layer_0_far_bg.webp",
            "layer_1_deep_mid.webp",
            "layer_2_near_mid.webp",
            "layer_3_hero_fg.webp",
            "layer_4_front_fx.webp",
        )

    val clockOcclusionFile: String get() = "clock_occlusion_mask.webp"
    val depthMapFile: String get() = "depth_map.webp"
    val subjectMaskFile: String get() = "subject_mask.webp"
}
