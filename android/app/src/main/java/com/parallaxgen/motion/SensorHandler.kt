package com.parallaxgen.motion

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log

private const val TAG = "SensorHandler"

/**
 * Reads device orientation via TYPE_ROTATION_VECTOR (gyro + accel fusion)
 * and falls back to TYPE_ACCELEROMETER for emulators without gyroscope.
 *
 * Outputs normalized tilt in [-1, 1] via [TiltListener].
 */
class SensorHandler(context: Context) : SensorEventListener {

    fun interface TiltListener {
        fun onTiltChanged(tiltX: Float, tiltY: Float)
    }

    var listener: TiltListener? = null

    /** Multiplier applied to raw orientation radians before clamping. */
    var sensitivity: Float = 2.5f

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val rotationVectorSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private val accelerometerSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val rotationMatrix = FloatArray(9)
    private val orientation = FloatArray(3)

    /** True when using rotation vector, false when falling back to accelerometer. */
    val hasGyroscope: Boolean get() = rotationVectorSensor != null

    fun start() {
        val sensor = rotationVectorSensor ?: accelerometerSensor
        if (sensor != null) {
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "Registered ${sensor.name}")
        } else {
            Log.w(TAG, "No motion sensor available — parallax disabled")
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    // ---- SensorEventListener ------------------------------------------------

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> handleRotationVector(event)
            Sensor.TYPE_ACCELEROMETER -> handleAccelerometer(event)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* unused */ }

    // ---- Internal -----------------------------------------------------------

    private fun handleRotationVector(event: SensorEvent) {
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
        SensorManager.getOrientation(rotationMatrix, orientation)
        // orientation[1] = pitch (forward/back), orientation[2] = roll (left/right)
        val tiltX = (orientation[2] * sensitivity).coerceIn(-1f, 1f)
        val tiltY = (orientation[1] * sensitivity).coerceIn(-1f, 1f)
        listener?.onTiltChanged(tiltX, tiltY)
    }

    private fun handleAccelerometer(event: SensorEvent) {
        // Emulator fallback: derive tilt from gravity direction
        val g = SensorManager.GRAVITY_EARTH
        val tiltX = (event.values[0] / g * sensitivity).coerceIn(-1f, 1f)
        val tiltY = (-event.values[1] / g * sensitivity).coerceIn(-1f, 1f)
        listener?.onTiltChanged(tiltX, tiltY)
    }
}
