package com.parallaxgen.corpus

import java.io.File

class CorpusLoader {
    fun loadPackage(root: File): WallpaperMeta {
        return WallpaperMeta(id = root.name)
    }
}
