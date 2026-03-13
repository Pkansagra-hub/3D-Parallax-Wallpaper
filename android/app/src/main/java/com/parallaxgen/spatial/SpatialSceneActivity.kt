package com.parallaxgen.spatial

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.parallaxgen.corpus.CorpusManager
import com.parallaxgen.motion.MotionController
import com.parallaxgen.motion.SensorHandler
import com.parallaxgen.renderer.ParallaxGLSurfaceView

private const val TAG = "SpatialSceneActivity"

class SpatialSceneActivity : ComponentActivity() {

    private var glSurfaceView: ParallaxGLSurfaceView? = null
    private var sensorHandler: SensorHandler? = null
    private val motionController = MotionController()
    private var lastSensorTimestamp = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load the first available wallpaper package
        val corpusManager = CorpusManager()
        val packages = corpusManager.loadCorpus(this)
        val selectedId = intent.getStringExtra("wallpaper_id")
        val pkg = if (selectedId != null) {
            packages.find { it.meta.id == selectedId }
        } else {
            packages.firstOrNull()
        }

        Log.i(TAG, "Available packages: ${packages.size}, selected: ${pkg?.meta?.id}")

        // Configure per-layer spring physics from wallpaper metadata
        if (pkg != null) {
            motionController.configure(pkg.meta)
        }

        // Set up gyroscope / accelerometer sensor
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
                    // GL Surface
                    AndroidView(
                        factory = { ctx ->
                            ParallaxGLSurfaceView(ctx).also { view ->
                                glSurfaceView = view
                                if (pkg != null) {
                                    view.parallaxRenderer.setWallpaperPackage(pkg)
                                }
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                    )

                    // Wallpaper title overlay
                    val title = remember { pkg?.title ?: "No wallpaper loaded" }
                    Text(
                        text = title,
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .padding(24.dp),
                    )
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
}
