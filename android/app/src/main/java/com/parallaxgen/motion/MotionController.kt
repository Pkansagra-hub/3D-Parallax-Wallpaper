package com.parallaxgen.motion

data class MotionState(
    val tiltX: Float = 0f,
    val tiltY: Float = 0f,
)

class MotionController(
    private val springX: DampedSpring = DampedSpring(),
    private val springY: DampedSpring = DampedSpring(),
) {
    private var state = MotionState()

    fun update(rawTiltX: Float, rawTiltY: Float, deltaSeconds: Float): MotionState {
        state = MotionState(
            tiltX = springX.step(state.tiltX, rawTiltX, deltaSeconds),
            tiltY = springY.step(state.tiltY, rawTiltY, deltaSeconds),
        )
        return state
    }
}
