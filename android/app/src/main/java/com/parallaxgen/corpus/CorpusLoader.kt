package com.parallaxgen.corpus

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

private const val TAG = "CorpusLoader"

class CorpusLoader {

    /** Load a single wallpaper package from a filesystem directory containing meta.json. */
    fun loadPackage(dir: File): WallpaperMeta? {
        val metaFile = File(dir, "meta.json")
        if (!metaFile.exists()) {
            Log.w(TAG, "Skipping ${dir.name}: no meta.json")
            return null
        }
        return try {
            val json = JSONObject(metaFile.readText())
            WallpaperMeta.fromJson(json)
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
            WallpaperMeta.fromJson(json)
        } catch (e: Exception) {
            Log.w(TAG, "Skipping asset $wallpaperId: ${e.message}")
            null
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
