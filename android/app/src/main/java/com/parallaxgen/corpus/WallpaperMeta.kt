package com.parallaxgen.corpus

import org.json.JSONObject

data class QualityInfo(
    val depthSeparation: Float = 0f,
    val maskCleanliness: Float = 0f,
    val clockReadability: Float = 0f,
    val warnings: List<String> = emptyList(),
    val passed: Boolean = false,
)

data class WallpaperMeta(
    val id: String,
    val version: Int = 2,
    val targetDevice: String = "Galaxy S26 Ultra",
    val resolution: List<Int> = listOf(1440, 3120),
    val layerCount: Int = 5,
    val clockPlaneIndex: Int = 3,
    val parallaxStrength: Float = 0.65f,
    val overscan: Float = 0.18f,
    val motionProfile: String = "cinematic_slow",
    val depthWeights: List<Float> = listOf(0.08f, 0.18f, 0.32f, 0.48f, 0.62f),
    val blurPx: List<Float> = listOf(1.2f, 0.8f, 0.3f, 0f, 0f),
    val clockWeight: Float = 0.24f,
    val clockFontScale: Float = 0.62f,
    val clockAnchor: List<Float> = listOf(0.5f, 0.22f),
    val safeClockRect: List<Float> = listOf(0.16f, 0.07f, 0.84f, 0.30f),
    val hasClockOcclusion: Boolean = true,
    val focusAnchor: List<Float> = listOf(0.5f, 0.36f),
    val subjectBbox: List<Float> = listOf(0f, 0f, 1f, 1f),
    val inpainted: Boolean = true,
    val depthModel: String = "depth_anything_v2_large",
    val segmentationModel: String = "birefnet",
    val quality: QualityInfo = QualityInfo(),
) {
    companion object {
        fun fromJson(json: JSONObject): WallpaperMeta {
            val resolution = json.optJSONArray("resolution")?.let { arr ->
                (0 until arr.length()).map { arr.getInt(it) }
            } ?: listOf(1440, 3120)

            val depthWeights = json.optJSONArray("depth_weights")?.let { arr ->
                (0 until arr.length()).map { arr.getDouble(it).toFloat() }
            } ?: listOf(0.08f, 0.18f, 0.32f, 0.48f, 0.62f)

            val blurPx = json.optJSONArray("blur_px")?.let { arr ->
                (0 until arr.length()).map { arr.getDouble(it).toFloat() }
            } ?: listOf(1.2f, 0.8f, 0.3f, 0f, 0f)

            val clockAnchor = json.optJSONArray("clock_anchor")?.let { arr ->
                (0 until arr.length()).map { arr.getDouble(it).toFloat() }
            } ?: listOf(0.5f, 0.22f)

            val safeClockRect = json.optJSONArray("safe_clock_rect")?.let { arr ->
                (0 until arr.length()).map { arr.getDouble(it).toFloat() }
            } ?: listOf(0.16f, 0.07f, 0.84f, 0.30f)

            val focusAnchor = json.optJSONArray("focus_anchor")?.let { arr ->
                (0 until arr.length()).map { arr.getDouble(it).toFloat() }
            } ?: listOf(0.5f, 0.36f)

            val subjectBbox = json.optJSONArray("subject_bbox")?.let { arr ->
                (0 until arr.length()).map { arr.getDouble(it).toFloat() }
            } ?: listOf(0f, 0f, 1f, 1f)

            val qualityObj = json.optJSONObject("quality")
            val quality = if (qualityObj != null) {
                val warnings = qualityObj.optJSONArray("warnings")?.let { arr ->
                    (0 until arr.length()).map { arr.getString(it) }
                } ?: emptyList()
                QualityInfo(
                    depthSeparation = qualityObj.optDouble("depth_separation", 0.0).toFloat(),
                    maskCleanliness = qualityObj.optDouble("mask_cleanliness", 0.0).toFloat(),
                    clockReadability = qualityObj.optDouble("clock_readability", 0.0).toFloat(),
                    warnings = warnings,
                    passed = qualityObj.optBoolean("passed", false),
                )
            } else {
                QualityInfo()
            }

            return WallpaperMeta(
                id = json.optString("id", "unknown"),
                version = json.optInt("version", 2),
                targetDevice = json.optString("target_device", "Galaxy S26 Ultra"),
                resolution = resolution,
                layerCount = json.optInt("layer_count", 5),
                clockPlaneIndex = json.optInt("clock_plane_index", 3),
                parallaxStrength = json.optDouble("parallax_strength", 0.65).toFloat(),
                overscan = json.optDouble("overscan", 0.18).toFloat(),
                motionProfile = json.optString("motion_profile", "cinematic_slow"),
                depthWeights = depthWeights,
                blurPx = blurPx,
                clockWeight = json.optDouble("clock_weight", 0.24).toFloat(),
                clockFontScale = json.optDouble("clock_font_scale", 0.62).toFloat(),
                clockAnchor = clockAnchor,
                safeClockRect = safeClockRect,
                hasClockOcclusion = json.optBoolean("has_clock_occlusion", true),
                focusAnchor = focusAnchor,
                subjectBbox = subjectBbox,
                inpainted = json.optBoolean("inpainted", true),
                depthModel = json.optString("depth_model", "depth_anything_v2_large"),
                segmentationModel = json.optString("segmentation_model", "birefnet"),
                quality = quality,
            )
        }
    }
}
