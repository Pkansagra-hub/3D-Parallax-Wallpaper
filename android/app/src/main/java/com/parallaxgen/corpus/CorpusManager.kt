package com.parallaxgen.corpus

import java.io.File

class CorpusManager(
    private val loader: CorpusLoader = CorpusLoader(),
) {
    fun listInstalledPackages(baseDir: File): List<WallpaperMeta> {
        if (!baseDir.exists()) {
            return emptyList()
        }
        return baseDir.listFiles()
            ?.filter(File::isDirectory)
            ?.map(loader::loadPackage)
            .orEmpty()
    }
}
