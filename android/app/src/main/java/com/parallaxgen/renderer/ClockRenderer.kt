package com.parallaxgen.renderer

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

class ClockRenderer {
    private val timeFormat = DateTimeFormatter.ofPattern("HH:mm")
    private val dateFormat = DateTimeFormatter.ofPattern("EEE, MMM d")

    fun renderSnapshot(now: LocalDateTime = LocalDateTime.now()): ClockSnapshot {
        return ClockSnapshot(
            timeText = now.format(timeFormat),
            dateText = now.format(dateFormat),
        )
    }
}

data class ClockSnapshot(
    val timeText: String,
    val dateText: String,
)
