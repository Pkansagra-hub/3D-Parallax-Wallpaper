package com.parallaxgen.wallpaper

import android.service.wallpaper.WallpaperService
import android.view.SurfaceHolder
import com.parallaxgen.corpus.WallpaperMeta

class ParallaxWallpaperService : WallpaperService() {
    override fun onCreateEngine(): Engine = EngineImpl()

    private class EngineImpl : Engine() {
        private val engine = ParallaxWallpaperEngine()

        override fun onVisibilityChanged(visible: Boolean) {
            if (visible) {
                engine.tick(WallpaperMeta(id = "starter"))
            }
        }

        override fun onSurfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            super.onSurfaceChanged(holder, format, width, height)
        }
    }
}
