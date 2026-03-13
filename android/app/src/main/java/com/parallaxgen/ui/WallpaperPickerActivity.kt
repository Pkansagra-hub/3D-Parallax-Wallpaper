package com.parallaxgen.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.parallaxgen.corpus.WallpaperMeta

class WallpaperPickerActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val meta = WallpaperMeta(id = "starter")
        setContent {
            MaterialTheme {
                Surface {
                    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
                        Text(text = "ParallaxGen", style = MaterialTheme.typography.headlineMedium)
                        Text(text = "Starter picker for spatial depth wallpapers")
                        WallpaperPreviewScreen(meta = meta)
                        Button(onClick = { }) {
                            Text(text = "Set Spatial Wallpaper")
                        }
                    }
                }
            }
        }
    }
}