package com.parallaxgen.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.parallaxgen.corpus.WallpaperMeta

@Composable
fun WallpaperPreviewScreen(meta: WallpaperMeta) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Text(text = meta.id, style = MaterialTheme.typography.headlineMedium)
        Text(text = "5-layer spatial preview with inserted clock plane")
    }
}
