package com.parallaxgen.renderer

object LayerShader {
    const val Vertex = "attribute vec4 aPosition; void main() { gl_Position = aPosition; }"
    const val Fragment = "precision mediump float; void main() { gl_FragColor = vec4(1.0); }"
}
