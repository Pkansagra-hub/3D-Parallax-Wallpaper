package com.parallaxgen.renderer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES30
import android.opengl.GLUtils
import android.util.Log
import com.parallaxgen.corpus.WallpaperPackage
import java.io.File

private const val TAG = "TextureLoader"

/** Max texture dimension — downscale if source is larger to save GPU memory. */
private const val MAX_TEXTURE_SIZE = 3120

/** Holds GL texture IDs for a loaded wallpaper scene. */
data class SceneTextures(
    val layers: List<Int>,           // 5 layer texture IDs (0–4)
    val clockOcclusion: Int,         // clock_occlusion_mask texture ID
    val preview: Int = 0,
) {
    fun release() {
        val all = layers + listOf(clockOcclusion, preview)
        val ids = all.filter { it != 0 }.toIntArray()
        if (ids.isNotEmpty()) {
            GLES30.glDeleteTextures(ids.size, ids, 0)
        }
    }
}

class TextureLoader {

    /** Load all scene textures for a wallpaper package. */
    fun loadScene(context: Context, pkg: WallpaperPackage): SceneTextures {
        val layerIds = pkg.layerFiles.map { filename ->
            loadTextureForPackage(context, pkg, filename)
        }
        val occlusionId = loadTextureForPackage(context, pkg, pkg.clockOcclusionFile)

        Log.i(TAG, "Loaded ${layerIds.count { it != 0 }}/5 layers + occlusion for ${pkg.meta.id}")
        return SceneTextures(layers = layerIds, clockOcclusion = occlusionId)
    }

    /** Load a single texture from a package (assets or filesystem). */
    private fun loadTextureForPackage(context: Context, pkg: WallpaperPackage, filename: String): Int {
        val bitmap = decodeBitmapForPackage(context, pkg, filename)
            ?: return 0
        return try {
            uploadTexture(bitmap)
        } catch (e: OutOfMemoryError) {
            Log.w(TAG, "OOM loading $filename — retrying at 50% scale")
            bitmap.recycle()
            val halfBitmap = decodeBitmapForPackage(context, pkg, filename, sampleSize = 2)
                ?: return 0
            try {
                uploadTexture(halfBitmap)
            } catch (e2: OutOfMemoryError) {
                Log.e(TAG, "OOM even at 50% — skipping $filename")
                halfBitmap.recycle()
                0
            }
        }
    }

    private fun decodeBitmapForPackage(
        context: Context,
        pkg: WallpaperPackage,
        filename: String,
        sampleSize: Int = 1,
    ): Bitmap? {
        return if (pkg.isAsset && pkg.assetBasePath != null) {
            decodeBitmapFromAssets(context, "${pkg.assetBasePath}/$filename", sampleSize)
        } else if (pkg.directory != null) {
            decodeBitmapFromFile(File(pkg.directory, filename), sampleSize)
        } else {
            null
        }
    }

    /** Decode bitmap from assets. */
    private fun decodeBitmapFromAssets(context: Context, path: String, sampleSize: Int = 1): Bitmap? {
        return try {
            val opts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            context.assets.open(path).use { stream ->
                BitmapFactory.decodeStream(stream, null, opts)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to decode asset: $path — ${e.message}")
            null
        }
    }

    /** Decode bitmap from filesystem. */
    private fun decodeBitmapFromFile(file: File, sampleSize: Int = 1): Bitmap? {
        if (!file.exists()) {
            Log.w(TAG, "File not found: ${file.absolutePath}")
            return null
        }
        return try {
            val opts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            BitmapFactory.decodeFile(file.absolutePath, opts)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to decode file: ${file.absolutePath} — ${e.message}")
            null
        }
    }

    /** Upload bitmap to GL texture and recycle. Returns texture ID or 0 on failure. */
    private fun uploadTexture(bitmap: Bitmap): Int {
        val texIds = IntArray(1)
        GLES30.glGenTextures(1, texIds, 0)
        val texId = texIds[0]
        if (texId == 0) {
            Log.e(TAG, "glGenTextures failed")
            bitmap.recycle()
            return 0
        }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

        bitmap.recycle()
        return texId
    }
}
