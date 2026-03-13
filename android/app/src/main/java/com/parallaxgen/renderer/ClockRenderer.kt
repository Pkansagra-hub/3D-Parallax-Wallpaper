package com.parallaxgen.renderer

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.opengl.GLES30
import android.opengl.GLUtils
import android.util.Log
import com.parallaxgen.corpus.WallpaperMeta
import com.parallaxgen.settings.WallpaperTuning
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

private const val TAG = "ClockRenderer"

/**
 * Renders current time + date to a GL texture via Android Canvas.
 * Re-renders only when the displayed minute changes.
 *
 * The texture is full-resolution (matching wallpaper), transparent everywhere
 * except where the clock glyphs are drawn.
 */
class ClockRenderer {

    private data class ClockLayout(
        val centerX: Float,
        val timeBaselineY: Float,
        val dateBaselineY: Float,
        val timeSizePx: Float,
        val dateSizePx: Float,
    )

    private val timeFormat24 = DateTimeFormatter.ofPattern("HH:mm")
    private val timeFormat12 = DateTimeFormatter.ofPattern("h:mm")
    private val dateFormat = DateTimeFormatter.ofPattern("EEE, MMM d", Locale.getDefault())
    private var tuning = WallpaperTuning()

    /** GL texture id for the current clock face. 0 = not yet created. */
    var textureId: Int = 0
        private set

    /** Minute value that is currently rendered. -1 = never rendered. */
    private var renderedMinute: Int = -1

    private var bitmap: Bitmap? = null
    private var canvas: Canvas? = null

    fun setTuning(tuning: WallpaperTuning) {
        if (this.tuning == tuning) return
        this.tuning = tuning
        renderedMinute = -1
    }

    private val timeStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
        style = Paint.Style.STROKE
        strokeWidth = 1.2f
        alpha = 26
    }

    private val timeFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        alpha = 108
    }

    private val timeHighlightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        alpha = 34
    }

    private val timeCorePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        alpha = 76
    }

    private val datePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
        alpha = 150
        letterSpacing = 0.06f
    }

    /**
     * Check if the clock needs updating and, if so, re-render.
     * Call once per frame from the GL thread.
     *
     * @return true if the texture was regenerated this frame.
     */
    fun updateIfNeeded(meta: WallpaperMeta): Boolean {
        val now = LocalDateTime.now()
        val currentMinute = now.hour * 60 + now.minute
        if (currentMinute == renderedMinute && textureId != 0) return false

        render(meta, now)
        renderedMinute = currentMinute
        return true
    }

    /** Release GL texture. Call on GL thread when scene is torn down. */
    fun release() {
        if (textureId != 0) {
            val ids = intArrayOf(textureId)
            GLES30.glDeleteTextures(1, ids, 0)
            textureId = 0
        }
        bitmap?.recycle()
        bitmap = null
        canvas = null
        renderedMinute = -1
    }

    // =========================================================================

    private fun render(meta: WallpaperMeta, now: LocalDateTime) {
        val w = meta.resolution.getOrElse(0) { 1440 }
        val h = meta.resolution.getOrElse(1) { 3120 }

        // Re-use bitmap if same size
        if (bitmap == null || bitmap!!.width != w || bitmap!!.height != h) {
            bitmap?.recycle()
            bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            canvas = Canvas(bitmap!!)
        }

        val bmp = bitmap!!
        val cvs = canvas!!

        // Clear to transparent
        bmp.eraseColor(Color.TRANSPARENT)

        val timeText = now.format(if (tuning.use24Hour) timeFormat24 else timeFormat12)
        val dateText = now.format(dateFormat)
        val layout = computeLayout(meta, w, h, timeText)

        timeStrokePaint.textSize = layout.timeSizePx
        timeStrokePaint.strokeWidth = (layout.timeSizePx * 0.006f).coerceIn(0.9f, 2.2f)
        timeFillPaint.textSize = layout.timeSizePx
        timeHighlightPaint.textSize = layout.timeSizePx
        timeCorePaint.textSize = layout.timeSizePx
        datePaint.textSize = layout.dateSizePx

        if (tuning.showDate) {
            cvs.drawText(dateText.uppercase(Locale.getDefault()), layout.centerX, layout.dateBaselineY, datePaint)
        }
        cvs.drawText(timeText, layout.centerX, layout.timeBaselineY - layout.timeSizePx * 0.012f, timeHighlightPaint)
        cvs.drawText(timeText, layout.centerX, layout.timeBaselineY, timeFillPaint)
        cvs.drawText(timeText, layout.centerX, layout.timeBaselineY, timeCorePaint)
        cvs.drawText(timeText, layout.centerX, layout.timeBaselineY, timeStrokePaint)

        // Upload to GL texture
        uploadToTexture(bmp)

        Log.d(TAG, "Clock rendered: $timeText  $dateText")
    }

    private fun uploadToTexture(bmp: Bitmap) {
        if (textureId == 0) {
            val ids = IntArray(1)
            GLES30.glGenTextures(1, ids, 0)
            textureId = ids[0]
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bmp, 0)
        } else {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLUtils.texSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, bmp)
        }
    }

    private fun computeLayout(
        meta: WallpaperMeta,
        width: Int,
        height: Int,
        timeText: String,
    ): ClockLayout {
        val rect = meta.safeClockRect.takeIf { it.size >= 4 } ?: listOf(0.14f, 0.09f, 0.86f, 0.40f)
        val left = rect[0].coerceIn(0f, 1f) * width
        val top = maxOf(rect[1].coerceIn(0f, 1f), 0.10f) * height
        val right = rect[2].coerceIn(0f, 1f) * width
        val bottom = maxOf(rect[3].coerceIn(0f, 1f), 0.42f) * height
        val availableWidth = (right - left).coerceAtLeast(width * 0.5f)
        val availableHeight = (bottom - top).coerceAtLeast(height * 0.26f)

        var timeSizePx = (availableHeight * 0.67f).coerceAtLeast(height * 0.145f)
        timeFillPaint.textSize = timeSizePx
        val measuredWidth = timeFillPaint.measureText(timeText).coerceAtLeast(1f)
        val widthScale = (availableWidth * 0.92f) / measuredWidth
        if (widthScale < 1f) {
            timeSizePx *= widthScale
        }

        val dateSizePx = timeSizePx * 0.13f
        val centerX = (left + right) * 0.5f
        val centerY = top + availableHeight * 0.72f
        val timeBaselineY = centerY - (timeFillPaint.descent() + timeFillPaint.ascent()) * 0.5f
        val dateBaselineY = top + availableHeight * 0.16f

        return ClockLayout(
            centerX = centerX,
            timeBaselineY = timeBaselineY,
            dateBaselineY = dateBaselineY,
            timeSizePx = timeSizePx,
            dateSizePx = dateSizePx,
        )
    }
}
