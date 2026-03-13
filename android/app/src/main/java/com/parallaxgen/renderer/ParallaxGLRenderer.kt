package com.parallaxgen.renderer

import android.content.Context
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.util.Log
import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.corpus.WallpaperPackage
import com.parallaxgen.motion.LayerMotionState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

private const val TAG = "ParallaxGLRenderer"

/**
 * Real OpenGL ES 3.0 renderer for the 5-layer parallax scene.
 *
 * Draw order (back to front):
 *   layer_0_far_bg → layer_1_deep_mid → layer_2_near_mid
 *   → (clock plane — Milestone 4)
 *   → layer_3_hero_fg → layer_4_front_fx
 *
 * Each layer is a fullscreen textured quad scaled by (1 + overscan) so
 * parallax offsets never reveal edges.
 */
class ParallaxGLRenderer(
    private val context: Context,
) : GLSurfaceView.Renderer {

    // --- Scene state (set from main thread before GL is ready) ---
    @Volatile var pendingPackage: WallpaperPackage? = null

    // --- GL resources (only touched on GL thread) ---
    private var program = 0
    private var aPosition = 0
    private var aTexCoord = 0
    private var uOffset = 0
    private var uScale = 0
    private var uTexture = 0

    private var quadVertexBuffer: FloatBuffer? = null
    private var quadTexCoordBuffer: FloatBuffer? = null

    private var sceneTextures: SceneTextures? = null
    private var sceneLayers: List<SceneLayer> = emptyList()
    private var meta: WallpaperMeta? = null

    // --- Per-layer motion offsets (updated from sensor thread) ---
    // Volatile reference swap ensures visibility across threads.
    @Volatile var layerOffsetsX: FloatArray = FloatArray(5)
    @Volatile var layerOffsetsY: FloatArray = FloatArray(5)

    /** Set per-layer offsets from [MotionController]. Thread-safe. */
    fun setLayerOffsets(offsets: List<LayerMotionState>) {
        val newX = FloatArray(offsets.size) { offsets[it].offsetX }
        val newY = FloatArray(offsets.size) { offsets[it].offsetY }
        layerOffsetsX = newX
        layerOffsetsY = newY
    }

    // =========================================================================
    // GLSurfaceView.Renderer callbacks (GL thread)
    // =========================================================================

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        program = LayerShader.createProgram()
        if (program == 0) {
            Log.e(TAG, "Failed to create shader program")
            return
        }

        aPosition = GLES30.glGetAttribLocation(program, "aPosition")
        aTexCoord = GLES30.glGetAttribLocation(program, "aTexCoord")
        uOffset = GLES30.glGetUniformLocation(program, "uOffset")
        uScale = GLES30.glGetUniformLocation(program, "uScale")
        uTexture = GLES30.glGetUniformLocation(program, "uTexture")

        initQuadBuffers()
        loadPendingScene()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        // Check if a new scene needs loading
        if (pendingPackage != null && sceneTextures == null) {
            loadPendingScene()
        }

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        val textures = sceneTextures ?: return
        val m = meta ?: return
        if (program == 0) return

        GLES30.glUseProgram(program)

        val overscanScale = 1f + m.overscan  // 1.18 default

        // Snapshot per-layer offsets (volatile reads)
        val ox = layerOffsetsX
        val oy = layerOffsetsY

        // Draw layers 0, 1, 2 (behind clock)
        for (i in 0..2) {
            val layer = sceneLayers.getOrNull(i) ?: continue
            if (layer.textureId == 0) continue
            drawLayer(layer.textureId, i, ox, oy, overscanScale)
        }

        // Clock plane will be inserted here in Milestone 4

        // Draw layers 3, 4 (in front of clock)
        for (i in 3..4) {
            val layer = sceneLayers.getOrNull(i) ?: continue
            if (layer.textureId == 0) continue
            drawLayer(layer.textureId, i, ox, oy, overscanScale)
        }
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /** Queue a wallpaper package for loading on next frame. Thread-safe. */
    fun setWallpaperPackage(pkg: WallpaperPackage) {
        pendingPackage = pkg
        // Release existing textures on next frame
        sceneTextures?.release()
        sceneTextures = null
        sceneLayers = emptyList()
        meta = null
    }

    // =========================================================================
    // Internal
    // =========================================================================

    private fun loadPendingScene() {
        val pkg = pendingPackage ?: return
        pendingPackage = null

        val loader = TextureLoader()
        val textures = loader.loadScene(context, pkg)
        val composer = SceneComposer()
        val layers = composer.compose(pkg.meta, textures)

        sceneTextures = textures
        sceneLayers = layers
        meta = pkg.meta

        Log.i(TAG, "Scene loaded: ${pkg.meta.id} — ${layers.size} layers")
    }

    private fun drawLayer(
        textureId: Int,
        layerIndex: Int,
        ox: FloatArray,
        oy: FloatArray,
        overscanScale: Float,
    ) {
        // Per-layer offsets already include depth-weight scaling from MotionController
        val layerOffsetX = if (layerIndex < ox.size) ox[layerIndex] else 0f
        val layerOffsetY = if (layerIndex < oy.size) oy[layerIndex] else 0f

        GLES30.glUniform2f(uOffset, layerOffsetX, layerOffsetY)
        GLES30.glUniform1f(uScale, overscanScale)
        GLES30.glUniform1i(uTexture, 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)

        // Bind vertex positions
        quadVertexBuffer?.position(0)
        GLES30.glVertexAttribPointer(aPosition, 2, GLES30.GL_FLOAT, false, 0, quadVertexBuffer)
        GLES30.glEnableVertexAttribArray(aPosition)

        // Bind texture coordinates
        quadTexCoordBuffer?.position(0)
        GLES30.glVertexAttribPointer(aTexCoord, 2, GLES30.GL_FLOAT, false, 0, quadTexCoordBuffer)
        GLES30.glEnableVertexAttribArray(aTexCoord)

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun initQuadBuffers() {
        // Fullscreen quad in NDC (-1 to +1)
        val vertices = floatArrayOf(
            -1f, -1f,  // bottom-left
             1f, -1f,  // bottom-right
            -1f,  1f,  // top-left
             1f,  1f,  // top-right
        )
        quadVertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(vertices)
            .also { it.position(0) }

        // Texture coords (flipped Y for Android bitmap convention)
        val texCoords = floatArrayOf(
            0f, 1f,   // bottom-left
            1f, 1f,   // bottom-right
            0f, 0f,   // top-left
            1f, 0f,   // top-right
        )
        quadTexCoordBuffer = ByteBuffer.allocateDirect(texCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(texCoords)
            .also { it.position(0) }
    }
}
