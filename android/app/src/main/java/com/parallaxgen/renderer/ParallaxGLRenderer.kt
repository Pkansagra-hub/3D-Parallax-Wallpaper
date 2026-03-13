package com.parallaxgen.renderer

import android.content.Context
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.util.Log
import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.corpus.WallpaperPackage
import com.parallaxgen.motion.LayerMotionState
import com.parallaxgen.settings.WallpaperTuning
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

    // Clock shader program (samples clock texture + occlusion mask)
    private var clockProgram = 0
    private var clockAPosition = 0
    private var clockATexCoord = 0
    private var clockUOffset = 0
    private var clockUScale = 0
    private var clockUTexture = 0
    private var clockUBackgroundTexture = 0
    private var clockUOcclusionMask = 0
    private var clockUTexelSize = 0
    private var clockUClockOpacity = 0
    private var clockUHasOcclusion = 0

    private var quadVertexBuffer: FloatBuffer? = null
    private var quadTexCoordBuffer: FloatBuffer? = null

    private var sceneTextures: SceneTextures? = null
    private var sceneLayers: List<SceneLayer> = emptyList()
    private var meta: WallpaperMeta? = null
    private var maxDepthWeight: Float = 0.62f
    private var motionScaleX: Float = 1f
    private var motionScaleY: Float = 1f
    private var surfaceWidth: Int = 0
    private var surfaceHeight: Int = 0
    private var clockBackgroundTextureId: Int = 0
    private var clockBackgroundFramebufferId: Int = 0
    private val pendingSceneReleases = mutableListOf<SceneTextures>()
    private var releaseClockTexturePending = false
    @Volatile private var pendingTuning = WallpaperTuning()
    private var appliedTuning = WallpaperTuning()

    private val clockRenderer = ClockRenderer()

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
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)

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

        // Clock occlusion shader
        clockProgram = LayerShader.createClockProgram()
        if (clockProgram != 0) {
            clockAPosition = GLES30.glGetAttribLocation(clockProgram, "aPosition")
            clockATexCoord = GLES30.glGetAttribLocation(clockProgram, "aTexCoord")
            clockUOffset = GLES30.glGetUniformLocation(clockProgram, "uOffset")
            clockUScale = GLES30.glGetUniformLocation(clockProgram, "uScale")
            clockUTexture = GLES30.glGetUniformLocation(clockProgram, "uTexture")
            clockUBackgroundTexture = GLES30.glGetUniformLocation(clockProgram, "uBackgroundTexture")
            clockUOcclusionMask = GLES30.glGetUniformLocation(clockProgram, "uOcclusionMask")
            clockUTexelSize = GLES30.glGetUniformLocation(clockProgram, "uTexelSize")
            clockUClockOpacity = GLES30.glGetUniformLocation(clockProgram, "uClockOpacity")
            clockUHasOcclusion = GLES30.glGetUniformLocation(clockProgram, "uHasOcclusion")
        }

        initQuadBuffers()
        clockRenderer.setTuning(appliedTuning)
        loadPendingScene()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceWidth = width
        surfaceHeight = height
        ensureClockBackgroundBuffer(width, height)
        GLES30.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        releasePendingResources()
        applyPendingTuning()

        // Check if a new scene needs loading
        if (pendingPackage != null && sceneTextures == null) {
            loadPendingScene()
        }

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        val textures = sceneTextures ?: return
        val m = meta ?: return
        if (program == 0) return

        GLES30.glUseProgram(program)

        val overscanScale = 1f + m.overscan
        val maxMotion = (m.parallaxStrength * maxDepthWeight).coerceAtLeast(0.001f)
        val motionBudget = m.overscan * 0.56f
        motionScaleX = motionBudget / maxMotion
        motionScaleY = motionScaleX * 0.52f

        // Snapshot per-layer offsets (volatile reads)
        val ox = layerOffsetsX
        val oy = layerOffsetsY

        if (clockBackgroundTextureId != 0 && clockBackgroundFramebufferId != 0) {
            captureClockBackground(ox, oy, overscanScale)
            drawCapturedBackground()
        } else {
            drawLayerGroup(listOf(0, 1, 2), 1, ox, oy, overscanScale)
        }

        // --- Clock plane (between layers 2 and 3) ---
        drawClockPlane(m, overscanScale)

        // Draw layers 3, 4 (in front of clock)
        drawLayerGroup(listOf(3, 4), 3, ox, oy, overscanScale)
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /** Queue a wallpaper package for loading on next frame. Thread-safe. */
    fun setWallpaperPackage(pkg: WallpaperPackage) {
        pendingPackage = pkg
        sceneTextures?.let { pendingSceneReleases += it }
        releaseClockTexturePending = true
        sceneTextures = null
        sceneLayers = emptyList()
        meta = null
    }

    fun setTuning(tuning: WallpaperTuning) {
        pendingTuning = tuning
    }

    // =========================================================================
    // Internal
    // =========================================================================

    private fun loadPendingScene() {
        val pkg = pendingPackage ?: return
        pendingPackage = null

        try {
            val loader = TextureLoader()
            val textures = loader.loadScene(context, pkg)
            val composer = SceneComposer()
            val layers = composer.compose(pkg.meta, textures)

            sceneTextures = textures
            sceneLayers = layers
            meta = pkg.meta
            maxDepthWeight = pkg.meta.depthWeights.maxOrNull() ?: 0.62f
            clockRenderer.setTuning(appliedTuning)

            Log.i(TAG, "Scene loaded: ${pkg.meta.id} \u2014 ${layers.size} layers")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load scene: ${pkg.meta.id}", e)
        }
    }

    /**
     * Draws the clock texture with occlusion mask between layers 2 and 3.
     * Uses the dedicated clock shader program that multiplies clock alpha
     * by (1 - mask), making the subject silhouette cut through the clock.
     */
    private fun drawClockPlane(m: WallpaperMeta, overscanScale: Float) {
        // Update clock texture if minute changed
        clockRenderer.updateIfNeeded(m)
        val clockTexId = clockRenderer.textureId
        if (clockTexId == 0) return

        val occlusionTexId = sceneTextures?.clockOcclusion ?: 0
        val useOcclusion = m.hasClockOcclusion && occlusionTexId != 0
        val canUseGlassProgram = clockProgram != 0 && clockBackgroundTextureId != 0

        // Clock parallax: derive UV offset from clock depth weight
        val rawClockX = (if (layerOffsetsX.isNotEmpty()) layerOffsetsX[0] else 0f) /
            (meta?.depthWeights?.getOrElse(0) { 0.08f } ?: 0.08f) * m.clockWeight
        val rawClockY = (if (layerOffsetsY.isNotEmpty()) layerOffsetsY[0] else 0f) /
            (meta?.depthWeights?.getOrElse(0) { 0.08f } ?: 0.08f) * m.clockWeight
        val clockOffsetX = (rawClockX * motionScaleX * 1.20f).coerceIn(-0.12f, 0.12f)
        val clockOffsetY = (rawClockY * motionScaleY * 1.10f).coerceIn(-0.08f, 0.08f)

        if (canUseGlassProgram) {
            GLES30.glUseProgram(clockProgram)
            GLES30.glUniform2f(clockUOffset, clockOffsetX, clockOffsetY)
            GLES30.glUniform1f(clockUScale, overscanScale)
            GLES30.glUniform2f(
                clockUTexelSize,
                1f / surfaceWidth.coerceAtLeast(1).toFloat(),
                1f / surfaceHeight.coerceAtLeast(1).toFloat(),
            )
            GLES30.glUniform1f(clockUClockOpacity, appliedTuning.clockOpacity.coerceIn(0f, 1f))
            GLES30.glUniform1f(clockUHasOcclusion, if (useOcclusion) 1f else 0f)

            // Texture unit 0: clock
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, clockTexId)
            GLES30.glUniform1i(clockUTexture, 0)

            // Texture unit 1: captured background
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, clockBackgroundTextureId)
            GLES30.glUniform1i(clockUBackgroundTexture, 1)

            // Texture unit 2: occlusion mask
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, occlusionTexId)
            GLES30.glUniform1i(clockUOcclusionMask, 2)

            // Draw quad
            quadVertexBuffer?.position(0)
            GLES30.glVertexAttribPointer(clockAPosition, 2, GLES30.GL_FLOAT, false, 0, quadVertexBuffer)
            GLES30.glEnableVertexAttribArray(clockAPosition)

            quadTexCoordBuffer?.position(0)
            GLES30.glVertexAttribPointer(clockATexCoord, 2, GLES30.GL_FLOAT, false, 0, quadTexCoordBuffer)
            GLES30.glEnableVertexAttribArray(clockATexCoord)

            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

            // Reset active texture to unit 0
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            // Restore layer shader for subsequent layers
            GLES30.glUseProgram(program)
        } else {
            // No occlusion mask — draw clock with regular layer shader
            GLES30.glUniform2f(uOffset, clockOffsetX, clockOffsetY)
            GLES30.glUniform1f(uScale, overscanScale)
            GLES30.glUniform1i(uTexture, 0)

            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, clockTexId)

            quadVertexBuffer?.position(0)
            GLES30.glVertexAttribPointer(aPosition, 2, GLES30.GL_FLOAT, false, 0, quadVertexBuffer)
            GLES30.glEnableVertexAttribArray(aPosition)

            quadTexCoordBuffer?.position(0)
            GLES30.glVertexAttribPointer(aTexCoord, 2, GLES30.GL_FLOAT, false, 0, quadTexCoordBuffer)
            GLES30.glEnableVertexAttribArray(aTexCoord)

            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        }
    }

    private fun captureClockBackground(
        ox: FloatArray,
        oy: FloatArray,
        overscanScale: Float,
    ) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, clockBackgroundFramebufferId)
        GLES30.glViewport(0, 0, surfaceWidth, surfaceHeight)
        GLES30.glClearColor(0f, 0f, 0f, 0f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(program)

        drawLayerGroup(listOf(0, 1, 2), 1, ox, oy, overscanScale)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, surfaceWidth, surfaceHeight)
    }

    private fun drawCapturedBackground() {
        GLES30.glUseProgram(program)
        GLES30.glUniform2f(uOffset, 0f, 0f)
        GLES30.glUniform1f(uScale, 1f)
        GLES30.glUniform1i(uTexture, 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, clockBackgroundTextureId)

        quadVertexBuffer?.position(0)
        GLES30.glVertexAttribPointer(aPosition, 2, GLES30.GL_FLOAT, false, 0, quadVertexBuffer)
        GLES30.glEnableVertexAttribArray(aPosition)

        quadTexCoordBuffer?.position(0)
        GLES30.glVertexAttribPointer(aTexCoord, 2, GLES30.GL_FLOAT, false, 0, quadTexCoordBuffer)
        GLES30.glEnableVertexAttribArray(aTexCoord)

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun drawLayer(
        textureId: Int,
        layerIndex: Int,
        ox: FloatArray,
        oy: FloatArray,
        overscanScale: Float,
    ) {
        val layerOffsetX = if (layerIndex < ox.size) ox[layerIndex] else 0f
        val layerOffsetY = if (layerIndex < oy.size) oy[layerIndex] else 0f
        drawLayerWithOffset(textureId, layerOffsetX, layerOffsetY, overscanScale)
    }

    private fun drawLayerGroup(
        layerIndices: List<Int>,
        anchorIndex: Int,
        ox: FloatArray,
        oy: FloatArray,
        overscanScale: Float,
    ) {
        if (layerIndices.isEmpty()) return

        val groupedOffsetX = if (anchorIndex < ox.size) ox[anchorIndex] else 0f
        val groupedOffsetY = if (anchorIndex < oy.size) oy[anchorIndex] else 0f

        for (layerIndex in layerIndices) {
            val layer = sceneLayers.getOrNull(layerIndex) ?: continue
            if (layer.textureId == 0) continue
            drawLayerWithOffset(layer.textureId, groupedOffsetX, groupedOffsetY, overscanScale)
        }
    }

    private fun drawLayerWithOffset(
        textureId: Int,
        layerOffsetX: Float,
        layerOffsetY: Float,
        overscanScale: Float,
    ) {
        val renderOffsetX = (layerOffsetX * motionScaleX).coerceIn(-0.10f, 0.10f)
        val renderOffsetY = (layerOffsetY * motionScaleY).coerceIn(-0.06f, 0.06f)

        GLES30.glUniform2f(uOffset, renderOffsetX, renderOffsetY)
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

    private fun releasePendingResources() {
        if (pendingSceneReleases.isNotEmpty()) {
            pendingSceneReleases.forEach { it.release() }
            pendingSceneReleases.clear()
        }
        if (releaseClockTexturePending) {
            clockRenderer.release()
            releaseClockTexturePending = false
        }
    }

    private fun applyPendingTuning() {
        if (appliedTuning == pendingTuning) return
        appliedTuning = pendingTuning
        clockRenderer.setTuning(appliedTuning)
    }

    private fun ensureClockBackgroundBuffer(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        if (clockBackgroundTextureId != 0 && clockBackgroundFramebufferId != 0) {
            val fbos = intArrayOf(clockBackgroundFramebufferId)
            val textures = intArrayOf(clockBackgroundTextureId)
            GLES30.glDeleteFramebuffers(1, fbos, 0)
            GLES30.glDeleteTextures(1, textures, 0)
            clockBackgroundFramebufferId = 0
            clockBackgroundTextureId = 0
        }

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        clockBackgroundTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, clockBackgroundTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA,
            width,
            height,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_UNSIGNED_BYTE,
            null,
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        clockBackgroundFramebufferId = fbos[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, clockBackgroundFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            clockBackgroundTextureId,
            0,
        )

        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Clock background framebuffer incomplete: $status")
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            val invalidFbos = intArrayOf(clockBackgroundFramebufferId)
            val invalidTextures = intArrayOf(clockBackgroundTextureId)
            GLES30.glDeleteFramebuffers(1, invalidFbos, 0)
            GLES30.glDeleteTextures(1, invalidTextures, 0)
            clockBackgroundFramebufferId = 0
            clockBackgroundTextureId = 0
            return
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }
}
