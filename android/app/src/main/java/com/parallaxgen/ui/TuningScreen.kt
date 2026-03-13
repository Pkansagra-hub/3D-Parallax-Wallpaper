package com.parallaxgen.ui

import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.parallaxgen.settings.PREFS_NAME
import com.parallaxgen.settings.WallpaperTuning
import com.parallaxgen.settings.loadWallpaperTuning
import com.parallaxgen.settings.tuningKeyPrefix

@Composable
fun TuningPanel(
    wallpaperId: String,
    defaultStrength: Float = 0.65f,
    onTuningChanged: ((WallpaperTuning) -> Unit)? = null,
) {
    val context = LocalContext.current
    val prefs = remember {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }
    val keyPrefix = tuningKeyPrefix(wallpaperId)
    val savedTuning = remember(wallpaperId) {
        loadWallpaperTuning(prefs, wallpaperId, defaultStrength)
    }

    var parallaxStrength by remember {
        mutableFloatStateOf(savedTuning.parallaxStrength)
    }
    var clockOpacity by remember {
        mutableFloatStateOf(savedTuning.clockOpacity)
    }
    var dampingLevel by remember {
        mutableFloatStateOf(savedTuning.damping)
    }
    var use24h by remember {
        mutableStateOf(savedTuning.use24Hour)
    }
    var showDate by remember {
        mutableStateOf(savedTuning.showDate)
    }

    fun save() {
        val tuning = WallpaperTuning(
            parallaxStrength = parallaxStrength,
            clockOpacity = clockOpacity,
            damping = dampingLevel,
            use24Hour = use24h,
            showDate = showDate,
        )
        prefs.edit()
            .putFloat("${keyPrefix}parallax_strength", tuning.parallaxStrength)
            .putFloat("${keyPrefix}clock_opacity", tuning.clockOpacity)
            .putFloat("${keyPrefix}damping", tuning.damping)
            .putBoolean("${keyPrefix}use_24h", tuning.use24Hour)
            .putBoolean("${keyPrefix}show_date", tuning.showDate)
            .apply()
        onTuningChanged?.invoke(tuning)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.7f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Tuning", style = MaterialTheme.typography.titleSmall, color = Color.White)

            // Parallax strength
            TuningSlider(
                label = "Parallax",
                value = parallaxStrength,
                onValueChange = { parallaxStrength = it; save() },
            )

            // Clock opacity
            TuningSlider(
                label = "Clock Opacity",
                value = clockOpacity,
                onValueChange = { clockOpacity = it; save() },
            )

            // Motion damping
            TuningSlider(
                label = "Damping",
                value = dampingLevel,
                onValueChange = { dampingLevel = it; save() },
            )

            // 24h toggle
            TuningToggle(
                label = "24-hour clock",
                checked = use24h,
                onCheckedChange = { use24h = it; save() },
            )

            // Show date toggle
            TuningToggle(
                label = "Show date",
                checked = showDate,
                onCheckedChange = { showDate = it; save() },
            )
        }
    }
}

@Composable
private fun TuningSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, color = Color.White, style = MaterialTheme.typography.bodySmall)
            Text(
                "%.0f%%".format(value * 100),
                color = Color.White.copy(alpha = 0.7f),
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..1f,
        )
    }
}

@Composable
private fun TuningToggle(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, color = Color.White, style = MaterialTheme.typography.bodySmall)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
