package com.parallaxgen.motion

import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.settings.WallpaperTuning

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
    private var motionWeights: List<Float> = emptyList()
    private var baseStiffness: List<Float> = emptyList()
    private var baseDamping: List<Float> = emptyList()
    private var parallaxStrength: Float = 0.65f
    private var baseParallaxStrength: Float = 0.65f
    private var offsets: List<LayerMotionState> = emptyList()

    /** Vertical parallax is intentionally subdued on tall phones. */
    private val verticalScale = 0.35f

    /**
     * Initialise per-layer springs from the wallpaper metadata.
     * Must be called on the same thread that calls [update].
     */
    fun configure(meta: WallpaperMeta, tuning: WallpaperTuning = WallpaperTuning()) {
        val n = meta.depthWeights.size
        weights = meta.depthWeights
        motionWeights = buildMotionWeights(meta.depthWeights)
        baseParallaxStrength = meta.parallaxStrength
        parallaxStrength = meta.parallaxStrength
        baseStiffness = List(n) { i ->
            val w = weights[i]
            120f + 200f * w
        }
        baseDamping = List(n) { i ->
            val w = weights[i]
            16f + 12f * w
        }

        // Per-layer spring tuning: far layers (low weight) → low stiffness = floatier
        springsX = List(n) { i ->
            DampedSpring(
                stiffness = baseStiffness[i],
                damping = baseDamping[i],
            )
        }
        springsY = List(n) { i ->
            DampedSpring(
                stiffness = baseStiffness[i],
                damping = baseDamping[i],
            )
        }
        offsets = List(n) { LayerMotionState() }
        applyTuning(tuning)
    }

    fun applyTuning(tuning: WallpaperTuning) {
        if (springsX.isEmpty() || springsY.isEmpty()) return

        val parallaxMultiplier = lerp(0.45f, 1.35f, tuning.parallaxStrength)
        parallaxStrength = (baseParallaxStrength * parallaxMultiplier).coerceIn(0.12f, 1.15f)

        val dampingMultiplier = lerp(0.7f, 1.45f, tuning.damping)
        for (i in springsX.indices) {
            val damping = baseDamping[i] * dampingMultiplier
            springsX[i].setParameters(baseStiffness[i], damping)
            springsY[i].setParameters(baseStiffness[i], damping)
        }
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
            val weight = motionWeights.getOrElse(i) { weights[i] }
            val targetX = rawTiltX * parallaxStrength * weight
            val targetY = rawTiltY * parallaxStrength * weight * verticalScale
            LayerMotionState(
                offsetX = springsX[i].step(offsets[i].offsetX, targetX, deltaSeconds),
                offsetY = springsY[i].step(offsets[i].offsetY, targetY, deltaSeconds),
            )
        }
        return offsets
    }

    private fun buildMotionWeights(depthWeights: List<Float>): List<Float> {
        if (depthWeights.isEmpty()) return emptyList()

        val minWeight = depthWeights.minOrNull() ?: return depthWeights
        val maxWeight = depthWeights.maxOrNull() ?: return depthWeights
        val spread = (maxWeight - minWeight).coerceAtLeast(0.0001f)

        return depthWeights.map { weight ->
            val normalized = ((weight - minWeight) / spread).coerceIn(0f, 1f)
            val compressed = normalized * normalized * 0.55f + normalized * 0.45f
            (0.12f + compressed * 0.20f).coerceIn(0.12f, 0.32f)
        }
    }

    private fun lerp(start: Float, end: Float, amount: Float): Float {
        return start + (end - start) * amount.coerceIn(0f, 1f)
    }
}
