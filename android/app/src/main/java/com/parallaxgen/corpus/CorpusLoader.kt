package com.parallaxgen.corpus

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONObject
import java.io.File

private const val TAG = "CorpusLoader"

class CorpusLoader {

    private val requiredLayerFiles = listOf(
        "layer_0_far_bg.webp",
        "layer_1_deep_mid.webp",
        "layer_2_near_mid.webp",
        "layer_3_hero_fg.webp",
        "layer_4_front_fx.webp",
        "clock_occlusion_mask.webp",
        "preview.webp",
    )

    private val minLayerBytesByFile = mapOf(
        "layer_0_far_bg.webp" to 24_000L,
        "layer_1_deep_mid.webp" to 12_000L,
        "layer_2_near_mid.webp" to 12_000L,
        "layer_3_hero_fg.webp" to 24_000L,
        "layer_4_front_fx.webp" to 2_000L,
        "clock_occlusion_mask.webp" to 1_000L,
        "preview.webp" to 40_000L,
    )

    /** Load a single wallpaper package from a filesystem directory containing meta.json. */
    fun loadPackage(dir: File): WallpaperMeta? {
        val metaFile = File(dir, "meta.json")
        if (!metaFile.exists()) {
            Log.w(TAG, "Skipping ${dir.name}: no meta.json")
            return null
        }
        return try {
            val json = JSONObject(metaFile.readText())
            val meta = WallpaperMeta.fromJson(json)
            if (isValidDirectoryPackage(dir, meta)) meta else null
        } catch (e: Exception) {
            Log.w(TAG, "Skipping ${dir.name}: ${e.message}")
            null
        }
    }

    /** Load a single wallpaper package from bundled assets. */
    fun loadPackageFromAssets(context: Context, wallpaperId: String): WallpaperMeta? {
        val path = "corpus/$wallpaperId/meta.json"
        return try {
            val text = context.assets.open(path).bufferedReader().readText()
            val json = JSONObject(text)
            val meta = WallpaperMeta.fromJson(json)
            if (isValidAssetPackage(context, wallpaperId, meta)) meta else null
        } catch (e: Exception) {
            Log.w(TAG, "Skipping asset $wallpaperId: ${e.message}")
            null
        }
    }

    private fun isValidDirectoryPackage(dir: File, meta: WallpaperMeta): Boolean {
        if (!passesMetaQualityGate(meta, dir.name)) return false
        for (name in requiredLayerFiles) {
            val file = File(dir, name)
            val minBytes = minLayerBytesByFile[name] ?: 1L
            if (!file.exists() || file.length() < minBytes) {
                Log.w(TAG, "Skipping ${dir.name}: invalid file $name (${file.length()} bytes)")
                return false
            }
            if (!hasRenderableImageBounds(file)) {
                Log.w(TAG, "Skipping ${dir.name}: unreadable image $name")
                return false
            }
        }
        return true
    }

    private fun isValidAssetPackage(context: Context, wallpaperId: String, meta: WallpaperMeta): Boolean {
        if (!passesMetaQualityGate(meta, wallpaperId)) return false
        for (name in requiredLayerFiles) {
            val assetPath = "corpus/$wallpaperId/$name"
            val bytes = try {
                context.assets.open(assetPath).use { it.available().toLong() }
            } catch (e: Exception) {
                Log.w(TAG, "Skipping asset $wallpaperId: missing $name")
                return false
            }
            val minBytes = minLayerBytesByFile[name] ?: 1L
            if (bytes < minBytes) {
                Log.w(TAG, "Skipping asset $wallpaperId: invalid file $name ($bytes bytes)")
                return false
            }
            if (!hasRenderableImageBounds(context, assetPath)) {
                Log.w(TAG, "Skipping asset $wallpaperId: unreadable image $name")
                return false
            }
        }
        return true
    }

    private fun passesMetaQualityGate(meta: WallpaperMeta, packageId: String): Boolean {
        if (meta.layerCount < 5 || meta.depthWeights.size < 5) {
            Log.w(TAG, "Skipping $packageId: incomplete layer metadata")
            return false
        }
        if (!meta.quality.passed) {
            Log.w(TAG, "Skipping $packageId: quality gate did not pass")
            return false
        }
        if (meta.quality.maskCleanliness < 0.70f) {
            Log.w(TAG, "Skipping $packageId: mask cleanliness too low (${meta.quality.maskCleanliness})")
            return false
        }
        return true
    }

    private fun hasRenderableImageBounds(file: File): Boolean {
        return try {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(file.absolutePath, opts)
            opts.outWidth >= 256 && opts.outHeight >= 256
        } catch (_: Exception) {
            false
        }
    }

    private fun hasRenderableImageBounds(context: Context, assetPath: String): Boolean {
        return try {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.assets.open(assetPath).use { stream ->
                BitmapFactory.decodeStream(stream, null, opts)
            }
            opts.outWidth >= 256 && opts.outHeight >= 256
        } catch (_: Exception) {
            false
        }
    }

    /** Parse index.json from a filesystem corpus directory. */
    fun loadIndex(corpusDir: File): List<CorpusEntry> {
        val indexFile = File(corpusDir, "index.json")
        if (!indexFile.exists()) return emptyList()
        return try {
            val json = JSONObject(indexFile.readText())
            val arr = json.getJSONArray("wallpapers")
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                CorpusEntry(
                    id = obj.getString("id"),
                    title = obj.getString("title"),
                    previewPath = obj.getString("preview"),
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse index.json: ${e.message}")
            emptyList()
        }
    }

    /** Parse index.json from bundled assets. */
    fun loadIndexFromAssets(context: Context): List<CorpusEntry> {
        return try {
            val text = context.assets.open("corpus/index.json").bufferedReader().readText()
            val json = JSONObject(text)
            val arr = json.getJSONArray("wallpapers")
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                CorpusEntry(
                    id = obj.getString("id"),
                    title = obj.getString("title"),
                    previewPath = obj.getString("preview"),
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse assets index.json: ${e.message}")
            emptyList()
        }
    }
}
