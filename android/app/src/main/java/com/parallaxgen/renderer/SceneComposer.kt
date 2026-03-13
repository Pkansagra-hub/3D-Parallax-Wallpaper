package com.parallaxgen.renderer

import com.parallaxgen.corpus.WallpaperMeta

data class SceneLayer(
    val name: String,
    val weight: Float,
)

class SceneComposer {
    fun compose(meta: WallpaperMeta): List<SceneLayer> {
        return listOf(
            SceneLayer("layer_0_far_bg", meta.depthWeights.getOrElse(0) { 0.08f }),
            SceneLayer("layer_1_deep_mid", meta.depthWeights.getOrElse(1) { 0.18f }),
            SceneLayer("layer_2_near_mid", meta.depthWeights.getOrElse(2) { 0.32f }),
            SceneLayer("clock_plane", meta.clockWeight),
            SceneLayer("layer_3_hero_fg", meta.depthWeights.getOrElse(3) { 0.48f }),
            SceneLayer("layer_4_front_fx", meta.depthWeights.getOrElse(4) { 0.62f }),
        )
    }
}
