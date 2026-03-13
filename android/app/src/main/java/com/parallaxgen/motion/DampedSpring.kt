package com.parallaxgen.motion

class DampedSpring(
    private val stiffness: Float = 180f,
    private val damping: Float = 22f,
) {
    private var velocity = 0f

    fun step(current: Float, target: Float, deltaSeconds: Float): Float {
        val acceleration = (target - current) * stiffness - velocity * damping
        velocity += acceleration * deltaSeconds
        return current + velocity * deltaSeconds
    }
}
