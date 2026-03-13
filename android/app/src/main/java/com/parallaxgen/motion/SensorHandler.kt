package com.parallaxgen.motion

class SensorHandler {
    fun normalizeTilt(rawX: Float, rawY: Float): Pair<Float, Float> {
        return rawX.coerceIn(-1f, 1f) to rawY.coerceIn(-1f, 1f)
    }
}
