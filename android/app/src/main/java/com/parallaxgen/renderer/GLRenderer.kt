package com.parallaxgen.renderer

import com.parallaxgen.motion.MotionState

class GLRenderer(
    private val sceneComposer: SceneComposer = SceneComposer(),
    private val clockRenderer: ClockRenderer = ClockRenderer(),
) {
    fun renderFrame(scene: RenderScene, motionState: MotionState): RenderFrame {
        val layers = sceneComposer.compose(scene.meta)
        val clock = clockRenderer.renderSnapshot()
        return RenderFrame(layers = layers, clock = clock, motionState = motionState)
    }
}

data class RenderScene(val meta: com.parallaxgen.corpus.WallpaperMeta)

data class RenderFrame(
    val layers: List<SceneLayer>,
    val clock: ClockSnapshot,
    val motionState: MotionState,
)
