package com.parallaxgen.corpus

data class WallpaperMeta(
    val id: String,
    val version: Int = 2,
    val resolution: List<Int> = listOf(1080, 2400),
    val layerCount: Int = 5,
    val clockPlaneIndex: Int = 3,
    val parallaxStrength: Float = 0.65f,
    val overscan: Float = 0.18f,
    val motionProfile: String = "cinematic_slow",
    val depthWeights: List<Float> = listOf(0.08f, 0.18f, 0.32f, 0.48f, 0.62f),
    val clockWeight: Float = 0.24f,
    val safeClockRect: List<Float> = listOf(0.16f, 0.07f, 0.84f, 0.30f),
)
