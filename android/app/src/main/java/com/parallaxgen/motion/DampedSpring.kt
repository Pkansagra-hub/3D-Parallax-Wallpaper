package com.parallaxgen.motion

class DampedSpring(
    private var stiffness: Float = 180f,
    private var damping: Float = 22f,
) {
    private var velocity = 0f

    fun setParameters(stiffness: Float, damping: Float) {
        this.stiffness = stiffness
        this.damping = damping
    }

    fun step(current: Float, target: Float, deltaSeconds: Float): Float {
        val acceleration = (target - current) * stiffness - velocity * damping
        velocity += acceleration * deltaSeconds
        return current + velocity * deltaSeconds
    }
}
