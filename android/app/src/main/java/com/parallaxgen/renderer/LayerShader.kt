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

    /**
     * Clock fragment shader: multiplies clock alpha by (1 - occlusion mask).
     * Where the mask is white (subject silhouette), the clock becomes transparent
     * so the hero layer drawn on top partially covers it.
     */
    val ClockFragment = """
        precision mediump float;
        uniform sampler2D uTexture;
        uniform sampler2D uBackgroundTexture;
        uniform sampler2D uOcclusionMask;
        uniform vec2 uTexelSize;
        uniform float uClockOpacity;
        uniform float uHasOcclusion;
        varying vec2 vTexCoord;

        vec3 sampleBlur(vec2 uv) {
            vec2 stepX = vec2(uTexelSize.x * 5.0, 0.0);
            vec2 stepY = vec2(0.0, uTexelSize.y * 5.0);
            vec3 color = texture2D(uBackgroundTexture, uv).rgb * 0.24;
            color += texture2D(uBackgroundTexture, uv + stepX).rgb * 0.12;
            color += texture2D(uBackgroundTexture, uv - stepX).rgb * 0.12;
            color += texture2D(uBackgroundTexture, uv + stepY).rgb * 0.12;
            color += texture2D(uBackgroundTexture, uv - stepY).rgb * 0.12;
            color += texture2D(uBackgroundTexture, uv + stepX + stepY).rgb * 0.10;
            color += texture2D(uBackgroundTexture, uv + stepX - stepY).rgb * 0.10;
            color += texture2D(uBackgroundTexture, uv - stepX + stepY).rgb * 0.10;
            color += texture2D(uBackgroundTexture, uv - stepX - stepY).rgb * 0.10;
            return color;
        }

        void main() {
            vec4 clock = texture2D(uTexture, vTexCoord);
            float mask = 0.0;
            if (uHasOcclusion > 0.5) {
                mask = texture2D(uOcclusionMask, vTexCoord).r;
            }
            float vis = 1.0 - mask;
            float glyph = clamp(clock.a, 0.0, 1.0);
            float shadowGlyph = texture2D(uTexture, vTexCoord + vec2(uTexelSize.x * 5.0, uTexelSize.y * 7.0)).a;
            float shadow = shadowGlyph * (1.0 - glyph) * 0.34 * vis;
            float alpha = glyph * vis * uClockOpacity * 0.62;
            vec3 blurred = sampleBlur(vTexCoord);
            float luma = dot(blurred, vec3(0.299, 0.587, 0.114));
            vec3 brightened = min(vec3(1.0), blurred * 1.03 + vec3(0.06));
            vec3 frosted = mix(brightened, vec3(luma), 0.30);
            float verticalGradient = smoothstep(0.95, 0.15, vTexCoord.y);
            vec3 gradientTint = mix(vec3(0.74, 0.82, 0.96), vec3(0.95, 0.97, 1.0), verticalGradient);
            vec3 glassBase = mix(frosted, gradientTint, 0.28);
            float edge = smoothstep(0.04, 0.30, glyph) - smoothstep(0.66, 0.99, glyph);
            float innerGlow = smoothstep(0.18, 0.92, glyph) * 0.08;
            vec3 glass = glassBase + vec3(edge * 0.09 + innerGlow) + vec3(0.012, 0.016, 0.02);
            vec3 shadowColor = vec3(0.02, 0.03, 0.06);
            gl_FragColor = vec4(shadowColor * shadow + glass * alpha, shadow + alpha);
        }
    """.trimIndent()

    /** Link vertex + clock-fragment shaders. Returns program ID or 0 on failure. */
    fun createClockProgram(): Int {
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, Vertex)
        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, ClockFragment)
        if (vertexShader == 0 || fragmentShader == 0) return 0

        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)

        val status = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(program)
            Log.e(TAG, "Clock program link failed: $log")
            GLES30.glDeleteProgram(program)
            return 0
        }
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)
        return program
    }

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
