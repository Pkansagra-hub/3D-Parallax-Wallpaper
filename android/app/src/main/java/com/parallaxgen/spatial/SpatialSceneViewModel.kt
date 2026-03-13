package com.parallaxgen.spatial

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import com.parallaxgen.corpus.CorpusManager
import com.parallaxgen.corpus.WallpaperPackage

private const val TAG = "SpatialSceneViewModel"

class SpatialSceneViewModel(app: Application) : AndroidViewModel(app) {

    private val corpusManager = CorpusManager()

    val packages: List<WallpaperPackage> by lazy {
        corpusManager.loadCorpus(getApplication()).also {
            Log.i(TAG, "Loaded ${it.size} wallpaper packages")
        }
    }

    val selectedPackage: WallpaperPackage?
        get() = packages.firstOrNull()
}
