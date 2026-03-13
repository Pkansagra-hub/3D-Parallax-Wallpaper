package com.parallaxgen.ui

import android.app.WallpaperManager
import android.content.ComponentName
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.parallaxgen.corpus.CorpusManager
import com.parallaxgen.corpus.WallpaperPackage
import com.parallaxgen.wallpaper.ParallaxWallpaperService
import java.io.File
import java.util.zip.ZipInputStream

private const val TAG = "WallpaperPicker"
private const val PREFS_NAME = "parallaxgen_prefs"
private const val KEY_SELECTED_WALLPAPER = "selected_wallpaper_id"

class WallpaperPickerActivity : ComponentActivity() {

    private val corpusManager = CorpusManager()
    private var packages by mutableStateOf<List<WallpaperPackage>>(emptyList())

    // File picker for .parallax.zip import
    private val importFileLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) importZipFile(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        reloadCorpus()

        setContent {
            MaterialTheme {
                PickerScreen(
                    packages = packages,
                    onSelectWallpaper = { pkg -> openPreview(pkg) },
                    onImportClick = {
                        importFileLauncher.launch(arrayOf("application/zip", "application/octet-stream"))
                    },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        reloadCorpus()
    }

    private fun reloadCorpus() {
        packages = corpusManager.loadCorpus(this)
    }

    private fun openPreview(pkg: WallpaperPackage) {
        val intent = Intent(this, WallpaperPreviewActivity::class.java)
            .putExtra("wallpaper_id", pkg.meta.id)
        startActivity(intent)
    }

    private fun importZipFile(uri: Uri) {
        try {
            val internalCorpus = corpusManager.getInternalCorpusDir(this)
            contentResolver.openInputStream(uri)?.use { input ->
                val zis = ZipInputStream(input)
                var entry = zis.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory) {
                        val outFile = File(internalCorpus, entry.name)
                        // Prevent zip-slip
                        if (!outFile.canonicalPath.startsWith(internalCorpus.canonicalPath)) {
                            Log.w(TAG, "Skipping zip entry outside target: ${entry.name}")
                            zis.closeEntry()
                            entry = zis.nextEntry
                            continue
                        }
                        outFile.parentFile?.mkdirs()
                        outFile.outputStream().use { out -> zis.copyTo(out) }
                    }
                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }
            reloadCorpus()
            Toast.makeText(this, "Wallpaper imported!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Import failed", e)
            Toast.makeText(this, "Import failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PickerScreen(
    packages: List<WallpaperPackage>,
    onSelectWallpaper: (WallpaperPackage) -> Unit,
    onImportClick: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("ParallaxGen") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                ),
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = onImportClick) {
                Text("+", style = MaterialTheme.typography.headlineSmall)
            }
        },
    ) { innerPadding ->
        if (packages.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding),
                contentAlignment = Alignment.Center,
            ) {
                Text("No wallpapers found.\nTap + to import.", style = MaterialTheme.typography.bodyLarge)
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                contentPadding = PaddingValues(
                    start = 12.dp, end = 12.dp,
                    top = innerPadding.calculateTopPadding() + 8.dp,
                    bottom = innerPadding.calculateBottomPadding() + 8.dp,
                ),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                items(packages, key = { it.meta.id }) { pkg ->
                    WallpaperCard(pkg = pkg, onClick = { onSelectWallpaper(pkg) })
                }
            }
        }
    }
}

@Composable
private fun WallpaperCard(pkg: WallpaperPackage, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Box {
            // Preview image
            val model = if (pkg.isAsset && pkg.assetBasePath != null) {
                ImageRequest.Builder(LocalContext.current)
                    .data("file:///android_asset/${pkg.assetBasePath}/preview.webp")
                    .crossfade(true)
                    .build()
            } else {
                ImageRequest.Builder(LocalContext.current)
                    .data(pkg.previewFile)
                    .crossfade(true)
                    .build()
            }
            AsyncImage(
                model = model,
                contentDescription = pkg.title,
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(9f / 16f)
                    .clip(RoundedCornerShape(12.dp)),
            )

            // Title overlay at bottom
            Text(
                text = pkg.title,
                color = Color.White,
                style = MaterialTheme.typography.labelLarge,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(8.dp),
            )
        }
    }
}
