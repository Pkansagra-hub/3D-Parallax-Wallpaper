package com.parallaxgen.corpus

/** Entry from index.json — lightweight reference for listing wallpapers. */
data class CorpusEntry(
    val id: String,
    val title: String,
    val previewPath: String,
)
