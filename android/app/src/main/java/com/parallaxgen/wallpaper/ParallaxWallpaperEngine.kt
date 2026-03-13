package com.parallaxgen.wallpaper

import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.motion.MotionController

/**
 * Stub engine — will be fully implemented in Milestone 5.
 * Currently just wires MotionController so the file compiles.
 */
class ParallaxWallpaperEngine {
    private val motionController = MotionController()

    fun tick(meta: WallpaperMeta) {
        motionController.configure(meta)
        motionController.update(rawTiltX = 0f, rawTiltY = 0f, deltaSeconds = 1f / 60f)
    }
}
