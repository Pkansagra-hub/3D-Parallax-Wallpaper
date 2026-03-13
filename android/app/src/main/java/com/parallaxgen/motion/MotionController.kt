package com.parallaxgen.motion

import com.parallaxgen.corpus.WallpaperMeta

/**
 * Per-layer offset produced by the motion system.
 * Values are in normalized offset space (roughly -0.05..+0.05 for typical motion).
 */
data class LayerMotionState(val offsetX: Float = 0f, val offsetY: Float = 0f)

/**
 * Produces per-layer parallax offsets by running independent [DampedSpring] pairs
 * for each layer.  Far-background layers use lower stiffness (more lag/float),
 * near layers track more closely to the raw tilt.
 *
 * Call [configure] once per wallpaper load, then [update] every sensor tick.
 */
class MotionController {

    private var springsX: List<DampedSpring> = emptyList()
    private var springsY: List<DampedSpring> = emptyList()
    private var weights: List<Float> = emptyList()
    private var parallaxStrength: Float = 0.65f
    private var offsets: List<LayerMotionState> = emptyList()

    /** Vertical parallax is 60% of horizontal to feel natural on tall phones. */
    private val verticalScale = 0.6f

    /**
     * Initialise per-layer springs from the wallpaper metadata.
     * Must be called on the same thread that calls [update].
     */
    fun configure(meta: WallpaperMeta) {
        val n = meta.depthWeights.size
        weights = meta.depthWeights
        parallaxStrength = meta.parallaxStrength

        // Per-layer spring tuning: far layers (low weight) → low stiffness = floatier
        springsX = List(n) { i ->
            val w = weights[i]
            DampedSpring(
                stiffness = 120f + 200f * w,   // 0.08→136, 0.48→216, 0.62→244
                damping = 16f + 12f * w,        // 0.08→17,  0.48→22,  0.62→23
            )
        }
        springsY = List(n) { i ->
            val w = weights[i]
            DampedSpring(
                stiffness = 120f + 200f * w,
                damping = 16f + 12f * w,
            )
        }
        offsets = List(n) { LayerMotionState() }
    }

    /**
     * Step all springs toward the current tilt and return per-layer offsets.
     *
     * @param rawTiltX  normalised tilt from sensor, [-1 .. 1]
     * @param rawTiltY  normalised tilt from sensor, [-1 .. 1]
     * @param deltaSeconds  time since last call (typically ~0.016–0.020)
     */
    fun update(rawTiltX: Float, rawTiltY: Float, deltaSeconds: Float): List<LayerMotionState> {
        if (springsX.isEmpty()) return offsets

        offsets = List(springsX.size) { i ->
            val targetX = rawTiltX * parallaxStrength * weights[i]
            val targetY = rawTiltY * parallaxStrength * weights[i] * verticalScale
            LayerMotionState(
                offsetX = springsX[i].step(offsets[i].offsetX, targetX, deltaSeconds),
                offsetY = springsY[i].step(offsets[i].offsetY, targetY, deltaSeconds),
            )
        }
        return offsets
    }
}
