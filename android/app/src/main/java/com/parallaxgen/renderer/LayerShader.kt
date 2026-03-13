package com.parallaxgen.renderer

import android.opengl.GLES30
import android.util.Log

private const val TAG = "LayerShader"

/**
 * GLSL shaders for parallax layer rendering.
 * Vertex shader applies per-layer offset + scale for overscan/parallax.
 * Fragment shader samples RGBA texture with alpha blending.
 */
object LayerShader {

    val Vertex = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform vec2 uOffset;
        uniform float uScale;
        varying vec2 vTexCoord;
        void main() {
            vec2 pos = aPosition.xy * uScale + uOffset;
            gl_Position = vec4(pos, 0.0, 1.0);
            vTexCoord = aTexCoord;
        }
    """.trimIndent()

    val Fragment = """
        precision mediump float;
        uniform sampler2D uTexture;
        varying vec2 vTexCoord;
        void main() {
            gl_FragColor = texture2D(uTexture, vTexCoord);
        }
    """.trimIndent()

    /** Compile a shader from source. Returns shader ID or 0 on failure. */
    fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val status = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(shader)
            Log.e(TAG, "Shader compile failed: $log")
            GLES30.glDeleteShader(shader)
            return 0
        }
        return shader
    }

    /** Link vertex + fragment shaders into a program. Returns program ID or 0 on failure. */
    fun createProgram(): Int {
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, Vertex)
        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, Fragment)
        if (vertexShader == 0 || fragmentShader == 0) return 0

        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)

        val status = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(program)
            Log.e(TAG, "Program link failed: $log")
            GLES30.glDeleteProgram(program)
            return 0
        }

        // Shaders can be detached after linking
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)
        return program
    }
}
