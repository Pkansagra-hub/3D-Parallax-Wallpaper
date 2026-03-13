package com.parallaxgen.corpus

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File

private const val TAG = "CorpusManager"

class CorpusManager(
    private val loader: CorpusLoader = CorpusLoader(),
) {
    /**
     * Load all wallpaper packages from both bundled assets and external storage.
     * Deduplicates by wallpaper ID (external storage wins on conflict).
     */
    fun loadCorpus(context: Context): List<WallpaperPackage> {
        val packages = mutableMapOf<String, WallpaperPackage>()

        // 1. Bundled assets
        for (pkg in loadFromAssets(context)) {
            packages[pkg.meta.id] = pkg
        }

        // 2. External storage — overwrites assets on ID conflict
        for (pkg in loadFromStorage(context)) {
            packages[pkg.meta.id] = pkg
        }

        Log.i(TAG, "Loaded ${packages.size} wallpaper packages")
        return packages.values.toList()
    }

    /** Load wallpapers bundled in app/src/main/assets/corpus/. */
    fun loadFromAssets(context: Context): List<WallpaperPackage> {
        val entries = loader.loadIndexFromAssets(context)
        return entries.mapNotNull { entry ->
            val meta = loader.loadPackageFromAssets(context, entry.id) ?: return@mapNotNull null
            WallpaperPackage(
                meta = meta,
                title = entry.title,
                directory = null,
                previewFile = null,
                isAsset = true,
                assetBasePath = "corpus/${entry.id}",
            )
        }
    }

    /** Load wallpapers from external storage at /sdcard/ParallaxGen/corpus/. */
    fun loadFromStorage(context: Context): List<WallpaperPackage> {
        val externalDir = File(
            Environment.getExternalStorageDirectory(),
            "ParallaxGen/corpus",
        )
        return loadFromDirectory(externalDir) + loadFromDirectory(getInternalCorpusDir(context))
    }

    /** Load wallpapers from a filesystem directory. */
    fun loadFromDirectory(corpusDir: File): List<WallpaperPackage> {
        if (!corpusDir.exists()) return emptyList()

        val entries = loader.loadIndex(corpusDir)
        val entryMap = entries.associateBy { it.id }

        val dirs = corpusDir.listFiles()?.filter(File::isDirectory) ?: return emptyList()
        return dirs.mapNotNull { dir ->
            val meta = loader.loadPackage(dir) ?: return@mapNotNull null
            val entry = entryMap[meta.id]
            WallpaperPackage(
                meta = meta,
                title = entry?.title ?: meta.id.replace('-', ' ')
                    .replaceFirstChar { it.uppercase() },
                directory = dir,
                previewFile = File(dir, "preview.webp").takeIf { it.exists() },
            )
        }
    }

    /** Internal app storage for imported wallpapers. */
    fun getInternalCorpusDir(context: Context): File {
        return File(context.filesDir, "corpus").also { it.mkdirs() }
    }
}
