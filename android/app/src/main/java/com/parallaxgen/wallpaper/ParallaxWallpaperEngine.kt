package com.parallaxgen.wallpaper

import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.motion.MotionController
import com.parallaxgen.renderer.GLRenderer
import com.parallaxgen.renderer.RenderScene

class ParallaxWallpaperEngine(
    private val renderer: GLRenderer = GLRenderer(),
    private val motionController: MotionController = MotionController(),
) {
    fun tick(meta: WallpaperMeta) {
        val motion = motionController.update(rawTiltX = 0f, rawTiltY = 0f, deltaSeconds = 1f / 60f)
        renderer.renderFrame(RenderScene(meta), motion)
    }
}
