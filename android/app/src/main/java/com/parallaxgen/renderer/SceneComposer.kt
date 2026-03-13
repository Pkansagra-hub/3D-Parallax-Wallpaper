package com.parallaxgen.renderer

import com.parallaxgen.corpus.WallpaperMeta

/** A renderable layer with its texture ID, parallax weight, and draw order position. */
data class SceneLayer(
    val name: String,
    val textureId: Int,
    val weight: Float,
)

/**
 * Builds the ordered render list from metadata + loaded textures.
 * Returns layers 0–4 (no clock plane here — clock is injected at draw time).
 */
class SceneComposer {

    fun compose(meta: WallpaperMeta, textures: SceneTextures): List<SceneLayer> {
        val names = listOf(
            "layer_0_far_bg",
            "layer_1_deep_mid",
            "layer_2_near_mid",
            "layer_3_hero_fg",
            "layer_4_front_fx",
        )
        return names.mapIndexed { i, name ->
            SceneLayer(
                name = name,
                textureId = textures.layers.getOrElse(i) { 0 },
                weight = meta.depthWeights.getOrElse(i) { 0f },
            )
        }
    }
}
