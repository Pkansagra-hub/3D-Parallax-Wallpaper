package com.parallaxgen.corpus

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File

private const val TAG = "CorpusManager"
private const val CURATED_LIMIT = 15

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
        val assetPackages = loadFromAssets(context)
        for (pkg in assetPackages) {
            packages[pkg.meta.id] = pkg
        }

        // 2. External storage — overwrites assets on ID conflict
        val storagePackages = loadFromStorage(context)
        for (pkg in storagePackages) {
            packages[pkg.meta.id] = pkg
        }

        val rankedPackages = packages.values
            .sortedWith(
                compareByDescending<WallpaperPackage> { curationScore(it) }
                    .thenByDescending { it.meta.quality.maskCleanliness }
                    .thenByDescending { it.meta.quality.depthSeparation }
                    .thenByDescending { it.meta.quality.clockReadability }
                    .thenBy { it.title },
            )
        val curatedPackages = rankedPackages.take(CURATED_LIMIT)
        Log.i(
            TAG,
            "Loaded ${packages.size} wallpaper packages (assets=${assetPackages.size}, storage=${storagePackages.size}); curated=${curatedPackages.size} ids=${curatedPackages.joinToString { it.meta.id }}",
        )
        return curatedPackages
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
        val appExternalDir = getExternalCorpusDir(context)
        val legacyExternalDir = File(
            Environment.getExternalStorageDirectory(),
            "ParallaxGen/corpus",
        )
        val appExternalPackages = loadFromDirectory(appExternalDir)
        val internalPackages = loadFromDirectory(getInternalCorpusDir(context))
        val legacyPackages = loadFromDirectory(legacyExternalDir)
        Log.i(
            TAG,
            "Storage sources: appExternal=${appExternalDir.absolutePath} (${appExternalPackages.size}), internal=${getInternalCorpusDir(context).absolutePath} (${internalPackages.size}), legacy=${legacyExternalDir.absolutePath} (${legacyPackages.size})",
        )
        return appExternalPackages + internalPackages + legacyPackages
    }

    /** Load wallpapers from a filesystem directory. */
    fun loadFromDirectory(corpusDir: File): List<WallpaperPackage> {
        if (!corpusDir.exists()) return emptyList()

        return try {
            val entries = loader.loadIndex(corpusDir)
            val entryMap = entries.associateBy { it.id }

            val dirs = corpusDir.listFiles()?.filter(File::isDirectory) ?: return emptyList()
            dirs.mapNotNull { dir ->
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
        } catch (e: Exception) {
            Log.w(TAG, "Skipping corpus dir ${corpusDir.absolutePath}: ${e.message}")
            emptyList()
        }
    }

    /** App-scoped external storage for bulk corpus sync without SAF/runtime permission issues. */
    fun getExternalCorpusDir(context: Context): File {
        val baseDir = context.getExternalFilesDir(null)
        return File(baseDir, "corpus").also { it.mkdirs() }
    }

    /** Internal app storage for imported wallpapers. */
    fun getInternalCorpusDir(context: Context): File {
        return File(context.filesDir, "corpus").also { it.mkdirs() }
    }

    private fun curationScore(pkg: WallpaperPackage): Float {
        val quality = pkg.meta.quality
        val warningPenalty = (quality.warnings.size.coerceAtMost(3)) * 0.04f
        val idealDepthSeparation = 0.58f
        val depthDistancePenalty = kotlin.math.abs(quality.depthSeparation - idealDepthSeparation) * 0.22f
        return quality.maskCleanliness * 0.55f +
            quality.depthSeparation * 0.18f +
            quality.clockReadability * 0.15f -
            depthDistancePenalty -
            warningPenalty
    }
}
