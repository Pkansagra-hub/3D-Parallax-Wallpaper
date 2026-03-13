package com.parallaxgen.ui

import android.app.WallpaperManager
import android.content.ComponentName
import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.parallaxgen.corpus.CorpusManager
import com.parallaxgen.corpus.WallpaperPackage
import com.parallaxgen.motion.MotionController
import com.parallaxgen.motion.SensorHandler
import com.parallaxgen.renderer.ParallaxGLSurfaceView
import com.parallaxgen.settings.KEY_SELECTED_WALLPAPER
import com.parallaxgen.settings.KEY_WALLPAPER_APPLY_TOKEN
import com.parallaxgen.settings.PREFS_NAME
import com.parallaxgen.settings.WallpaperTuning
import com.parallaxgen.settings.loadWallpaperTuning
import com.parallaxgen.wallpaper.ParallaxWallpaperService

private const val TAG = "WallpaperPreview"

/**
 * Full-screen live GL preview of a selected wallpaper with parallax motion.
 * Overlay buttons: Set as Wallpaper, Tuning, Back.
 */
class WallpaperPreviewActivity : ComponentActivity() {

    private var glSurfaceView: ParallaxGLSurfaceView? = null
    private var sensorHandler: SensorHandler? = null
    private val motionController = MotionController()
    private var lastSensorTimestamp = 0L

    private var showTuning by mutableStateOf(false)

    private val prefs: SharedPreferences by lazy {
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val wallpaperId = intent.getStringExtra("wallpaper_id")
        val corpusManager = CorpusManager()
        val packages = corpusManager.loadCorpus(this)
        val pkg = packages.find { it.meta.id == wallpaperId } ?: packages.firstOrNull()
        val initialTuning = pkg?.let { loadWallpaperTuning(prefs, it.meta) } ?: WallpaperTuning()

        if (pkg != null) {
            motionController.configure(pkg.meta, initialTuning)
        }

        sensorHandler = SensorHandler(this).apply {
            listener = SensorHandler.TiltListener { tiltX, tiltY ->
                val now = SystemClock.elapsedRealtimeNanos()
                val dt = if (lastSensorTimestamp == 0L) {
                    1f / 60f
                } else {
                    ((now - lastSensorTimestamp) / 1_000_000_000f).coerceIn(0.001f, 0.1f)
                }
                lastSensorTimestamp = now
                val offsets = motionController.update(tiltX, tiltY, dt)
                glSurfaceView?.parallaxRenderer?.setLayerOffsets(offsets)
            }
        }

        setContent {
            MaterialTheme {
                Box(modifier = Modifier.fillMaxSize()) {
                    // Live GL preview
                    AndroidView(
                        factory = { ctx ->
                            ParallaxGLSurfaceView(ctx).also { view ->
                                glSurfaceView = view
                                if (pkg != null) {
                                    view.parallaxRenderer.setTuning(initialTuning)
                                    view.parallaxRenderer.setWallpaperPackage(pkg)
                                }
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                    )

                    // Bottom overlay controls
                    Column(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        // Title
                        Text(
                            text = pkg?.title ?: "No wallpaper",
                            color = Color.White,
                            style = MaterialTheme.typography.titleLarge,
                        )

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            OutlinedButton(
                                onClick = { finish() },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = Color.White,
                                ),
                            ) {
                                Text("Back")
                            }

                            OutlinedButton(
                                onClick = { showTuning = !showTuning },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = Color.White,
                                ),
                            ) {
                                Text(if (showTuning) "Hide Tuning" else "Tuning")
                            }

                            Button(
                                onClick = { pkg?.let { setAsWallpaper(it) } },
                            ) {
                                Text("Set Wallpaper")
                            }
                        }

                        // Tuning panel
                        if (showTuning && pkg != null) {
                            TuningPanel(
                                wallpaperId = pkg.meta.id,
                                defaultStrength = pkg.meta.parallaxStrength,
                                onTuningChanged = { tuning -> applyPreviewTuning(tuning) },
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        glSurfaceView?.onResume()
        lastSensorTimestamp = 0L
        sensorHandler?.start()
    }

    override fun onPause() {
        super.onPause()
        sensorHandler?.stop()
        glSurfaceView?.onPause()
    }

    private fun setAsWallpaper(pkg: WallpaperPackage) {
        // Save selection to prefs
        prefs.edit()
            .putString(KEY_SELECTED_WALLPAPER, pkg.meta.id)
            .putLong(KEY_WALLPAPER_APPLY_TOKEN, SystemClock.elapsedRealtime())
            .apply()

        // Launch system wallpaper picker for our service
        try {
            val intent = Intent(WallpaperManager.ACTION_CHANGE_LIVE_WALLPAPER).apply {
                putExtra(
                    WallpaperManager.EXTRA_LIVE_WALLPAPER_COMPONENT,
                    ComponentName(this@WallpaperPreviewActivity, ParallaxWallpaperService::class.java),
                )
            }
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch wallpaper picker", e)
            Toast.makeText(this, "Could not set wallpaper", Toast.LENGTH_SHORT).show()
        }
    }

    private fun applyPreviewTuning(tuning: WallpaperTuning) {
        motionController.applyTuning(tuning)
        glSurfaceView?.parallaxRenderer?.setTuning(tuning)
    }
}
