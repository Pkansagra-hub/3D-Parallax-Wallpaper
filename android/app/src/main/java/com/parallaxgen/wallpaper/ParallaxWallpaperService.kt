package com.parallaxgen.wallpaper

import android.content.SharedPreferences
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.os.SystemClock
import android.service.wallpaper.WallpaperService
import android.util.Log
import android.view.Choreographer
import android.view.SurfaceHolder
import com.parallaxgen.corpus.CorpusManager
import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.motion.MotionController
import com.parallaxgen.motion.SensorHandler
import com.parallaxgen.renderer.ParallaxGLRenderer
import com.parallaxgen.settings.KEY_SELECTED_WALLPAPER
import com.parallaxgen.settings.KEY_WALLPAPER_APPLY_TOKEN
import com.parallaxgen.settings.PREFS_NAME
import com.parallaxgen.settings.loadWallpaperTuning
import com.parallaxgen.settings.tuningKeyPrefix

private const val TAG = "ParallaxWallpaperSvc"

class ParallaxWallpaperService : WallpaperService() {
    override fun onCreateEngine(): Engine = EngineImpl()

    private inner class EngineImpl : Engine(), Choreographer.FrameCallback {

        // --- EGL state ---
        private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
        private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
        private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
        private var eglConfig: EGLConfig? = null

        // --- Rendering state (GL thread = main thread for WallpaperService) ---
        private var renderer: ParallaxGLRenderer? = null
        private var isVisible = false
        private var surfaceWidth = 0
        private var surfaceHeight = 0

        // --- Motion ---
        private var sensorHandler: SensorHandler? = null
        private val motionController = MotionController()
        private var lastSensorTimestamp = 0L
        private var currentMeta: WallpaperMeta? = null

        // --- Preferences ---
        private val prefs: SharedPreferences by lazy {
            getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        }
        private val prefsListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
            when {
                key == KEY_SELECTED_WALLPAPER || key == KEY_WALLPAPER_APPLY_TOKEN -> {
                    reloadWallpaper(forceSceneReload = true)
                }

                currentMeta?.id != null && key?.startsWith(tuningKeyPrefix(currentMeta!!.id)) == true -> {
                    applyCurrentTuning()
                }
            }
        }

        // =============================================================
        // Engine lifecycle
        // =============================================================

        override fun onCreate(surfaceHolder: SurfaceHolder) {
            super.onCreate(surfaceHolder)
            setTouchEventsEnabled(false)
            prefs.registerOnSharedPreferenceChangeListener(prefsListener)

            sensorHandler = SensorHandler(this@ParallaxWallpaperService).apply {
                listener = SensorHandler.TiltListener { tiltX, tiltY ->
                    val now = SystemClock.elapsedRealtimeNanos()
                    val dt = if (lastSensorTimestamp == 0L) {
                        1f / 60f
                    } else {
                        ((now - lastSensorTimestamp) / 1_000_000_000f).coerceIn(0.001f, 0.1f)
                    }
                    lastSensorTimestamp = now
                    val offsets = motionController.update(tiltX, tiltY, dt)
                    renderer?.setLayerOffsets(offsets)
                }
            }
        }

        override fun onSurfaceCreated(holder: SurfaceHolder) {
            super.onSurfaceCreated(holder)
            initEGL(holder)
        }

        override fun onSurfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            super.onSurfaceChanged(holder, format, width, height)
            surfaceWidth = width
            surfaceHeight = height
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                makeCurrent()
                GLES30.glViewport(0, 0, width, height)
            }
        }

        override fun onVisibilityChanged(visible: Boolean) {
            isVisible = visible
            if (visible) {
                lastSensorTimestamp = 0L
                sensorHandler?.start()
                Choreographer.getInstance().postFrameCallback(this)
            } else {
                sensorHandler?.stop()
                Choreographer.getInstance().removeFrameCallback(this)
            }
        }

        override fun onSurfaceDestroyed(holder: SurfaceHolder) {
            super.onSurfaceDestroyed(holder)
            isVisible = false
            Choreographer.getInstance().removeFrameCallback(this)
            sensorHandler?.stop()
            tearDownEGL()
        }

        override fun onDestroy() {
            prefs.unregisterOnSharedPreferenceChangeListener(prefsListener)
            super.onDestroy()
            tearDownEGL()
        }

        // =============================================================
        // Choreographer frame callback (vsync-driven 60fps)
        // =============================================================

        override fun doFrame(frameTimeNanos: Long) {
            if (!isVisible) return

            if (eglContext == EGL14.EGL_NO_CONTEXT) {
                Choreographer.getInstance().postFrameCallback(this)
                return
            }

            makeCurrent()
            renderer?.onDrawFrame(null)
            if (!EGL14.eglSwapBuffers(eglDisplay, eglSurface)) {
                Log.w(TAG, "eglSwapBuffers failed")
            }

            // Schedule next frame
            Choreographer.getInstance().postFrameCallback(this)
        }

        // =============================================================
        // EGL management
        // =============================================================

        private fun initEGL(holder: SurfaceHolder) {
            eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
                Log.e(TAG, "eglGetDisplay failed")
                return
            }

            val version = IntArray(2)
            if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
                Log.e(TAG, "eglInitialize failed")
                return
            }

            val configAttribs = intArrayOf(
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT or 0x0040, // ES3
                EGL14.EGL_NONE,
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)
            eglConfig = configs[0]
            if (eglConfig == null) {
                Log.e(TAG, "eglChooseConfig failed")
                return
            }

            val contextAttribs = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE,
            )
            eglContext = EGL14.eglCreateContext(
                eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0,
            )

            val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
            eglSurface = EGL14.eglCreateWindowSurface(
                eglDisplay, eglConfig, holder.surface, surfaceAttribs, 0,
            )

            makeCurrent()

            // Create and initialize the renderer on this EGL context
            renderer = ParallaxGLRenderer(this@ParallaxWallpaperService).also { r ->
                r.onSurfaceCreated(null, null)
                if (surfaceWidth > 0 && surfaceHeight > 0) {
                    r.onSurfaceChanged(null, surfaceWidth, surfaceHeight)
                }
                loadSelectedWallpaper(r)
            }

            Log.i(TAG, "EGL initialized — ES 3.0")
        }

        private fun makeCurrent() {
            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
        }

        private fun tearDownEGL() {
            renderer = null
            currentMeta = null
            if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(
                    eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT,
                )
                if (eglSurface != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglDestroySurface(eglDisplay, eglSurface)
                }
                if (eglContext != EGL14.EGL_NO_CONTEXT) {
                    EGL14.eglDestroyContext(eglDisplay, eglContext)
                }
                EGL14.eglTerminate(eglDisplay)
            }
            eglDisplay = EGL14.EGL_NO_DISPLAY
            eglContext = EGL14.EGL_NO_CONTEXT
            eglSurface = EGL14.EGL_NO_SURFACE
            eglConfig = null
        }

        // =============================================================
        // Wallpaper loading
        // =============================================================

        private fun loadSelectedWallpaper(r: ParallaxGLRenderer) {
            val corpusManager = CorpusManager()
            val packages = corpusManager.loadCorpus(this@ParallaxWallpaperService)
            if (packages.isEmpty()) {
                Log.w(TAG, "No wallpaper packages found")
                return
            }

            val selectedId = prefs.getString(KEY_SELECTED_WALLPAPER, null)
            val pkg = if (selectedId != null) {
                packages.find { it.meta.id == selectedId } ?: packages.first()
            } else {
                packages.first()
            }

            val tuning = loadWallpaperTuning(prefs, pkg.meta)
            currentMeta = pkg.meta
            motionController.configure(pkg.meta, tuning)
            r.setTuning(tuning)
            r.setWallpaperPackage(pkg)
            Log.i(TAG, "Wallpaper loaded: ${pkg.meta.id}")
        }

        private fun reloadWallpaper(forceSceneReload: Boolean) {
            val r = renderer ?: return
            if (eglContext == EGL14.EGL_NO_CONTEXT) return
            makeCurrent()
            if (forceSceneReload) {
                loadSelectedWallpaper(r)
            } else {
                applyCurrentTuning()
            }
        }

        private fun applyCurrentTuning() {
            val meta = currentMeta ?: return
            val r = renderer ?: return
            val tuning = loadWallpaperTuning(prefs, meta)
            motionController.applyTuning(tuning)
            r.setTuning(tuning)
        }
    }
}
