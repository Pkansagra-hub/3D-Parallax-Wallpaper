package com.parallaxgen.settings

import android.content.SharedPreferences
import com.parallaxgen.corpus.WallpaperMeta

const val PREFS_NAME = "parallaxgen_prefs"
const val KEY_SELECTED_WALLPAPER = "selected_wallpaper_id"
const val KEY_WALLPAPER_APPLY_TOKEN = "wallpaper_apply_token"

data class WallpaperTuning(
    val parallaxStrength: Float = 0.65f,
    val clockOpacity: Float = 0.9f,
    val damping: Float = 0.5f,
    val use24Hour: Boolean = true,
    val showDate: Boolean = true,
)

fun tuningKeyPrefix(wallpaperId: String): String = "tuning_${wallpaperId}_"

fun loadWallpaperTuning(
    prefs: SharedPreferences,
    meta: WallpaperMeta,
): WallpaperTuning = loadWallpaperTuning(prefs, meta.id, meta.parallaxStrength)

fun loadWallpaperTuning(
    prefs: SharedPreferences,
    wallpaperId: String,
    defaultStrength: Float,
): WallpaperTuning {
    val prefix = tuningKeyPrefix(wallpaperId)
    return WallpaperTuning(
        parallaxStrength = prefs.getFloat("${prefix}parallax_strength", defaultStrength),
        clockOpacity = prefs.getFloat("${prefix}clock_opacity", 0.9f),
        damping = prefs.getFloat("${prefix}damping", 0.5f),
        use24Hour = prefs.getBoolean("${prefix}use_24h", true),
        showDate = prefs.getBoolean("${prefix}show_date", true),
    )
}
