# ParallaxGen ProGuard / R8 rules

# Keep WallpaperService (registered in manifest)
-keep class com.parallaxgen.wallpaper.ParallaxWallpaperService { *; }

# Keep data classes used with JSON parsing (org.json reflection-free, but keep for safety)
-keep class com.parallaxgen.corpus.WallpaperMeta { *; }
-keep class com.parallaxgen.corpus.WallpaperMeta$QualityInfo { *; }
-keep class com.parallaxgen.corpus.CorpusEntry { *; }
-keep class com.parallaxgen.corpus.WallpaperPackage { *; }

# Keep GLES classes
-keep class android.opengl.** { *; }

# Coil (image loader)
-dontwarn coil.**
-keep class coil.** { *; }

# Compose
-dontwarn androidx.compose.**
-keep class androidx.compose.** { *; }

# Android standard
-keepattributes *Annotation*
-keepattributes SourceFile,LineNumberTable
