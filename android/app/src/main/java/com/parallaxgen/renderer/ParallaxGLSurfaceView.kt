package com.parallaxgen.renderer

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.AttributeSet

/**
 * OpenGL ES 3.0 surface for rendering the parallax wallpaper scene.
 * Used inside SpatialSceneActivity and the wallpaper preview.
 */
class ParallaxGLSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : GLSurfaceView(context, attrs) {

    val parallaxRenderer: ParallaxGLRenderer

    init {
        setEGLContextClientVersion(3)
        // Request RGBA 8888 with alpha for layer blending
        setEGLConfigChooser(8, 8, 8, 8, 0, 0)
        parallaxRenderer = ParallaxGLRenderer(context)
        setRenderer(parallaxRenderer)
        // Render continuously for smooth parallax (60fps via vsync)
        renderMode = RENDERMODE_CONTINUOUSLY
    }
}
