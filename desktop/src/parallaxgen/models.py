from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PARALLAX_FORMAT_VERSION = 2
PARALLAX_CORPUS_FORMAT = "parallaxgen-corpus-v2"
TARGET_DEVICE_NAME = "Galaxy S26 Ultra"
TARGET_RENDER_RESOLUTION = (1440, 3120)
DEFAULT_DEPTH_WEIGHTS = [0.08, 0.18, 0.32, 0.48, 0.62]
DEFAULT_LAYER_BLUR = [1.2, 0.8, 0.3, 0.0, 0.0]
REQUIRED_SUPPORT_ASSETS = (
    "clock_occlusion_mask.webp",
    "subject_mask.webp",
    "depth_map.webp",
    "preview.webp",
)
OPTIONAL_SUPPORT_ASSETS: tuple[str, ...] = ()

# --- Clock rendering defaults (Android renderer consumes these) -----------
# The clock is NOT a fixed overlay.  It is a render plane whose size, position,
# and occlusion are driven by per-wallpaper metadata.  The Android renderer
# draws it *between* layer_2_near_mid and layer_3_hero_fg so that the hero
# subject can partially cover the digits — exactly like the Apple depth clock.
DEFAULT_CLOCK_FONT_SCALE = 0.62  # relative to viewport width
DEFAULT_CLOCK_ANCHOR = (0.50, 0.22)  # normalised (cx, cy) — centre of digits
DEFAULT_CLOCK_WEIGHT = 0.24  # parallax motion weight for the clock plane


@dataclass(frozen=True, slots=True)
class PackageContract:
    """Canonical on-disk contract for one generated wallpaper package.

    Required assets are always expected inside `<output>/<wallpaper_id>/`.
    Optional assets may be added later without changing the v2 package shape.
    """

    meta_filename: str = "meta.json"
    index_filename: str = "index.json"
    required_support_assets: tuple[str, ...] = REQUIRED_SUPPORT_ASSETS
    optional_support_assets: tuple[str, ...] = OPTIONAL_SUPPORT_ASSETS

    def required_asset_names(self, layers: list["LayerSpec"]) -> list[str]:
        return [layer.file_name for layer in layers] + list(
            self.required_support_assets
        )


PACKAGE_CONTRACT = PackageContract()


@dataclass(slots=True)
class LayerSpec:
    name: str
    asset_path: str
    weight: float
    blur_px: float = 0.0

    @property
    def file_name(self) -> str:
        return f"{self.name}.webp"


@dataclass(slots=True)
class WallpaperMeta:
    """Canonical ``meta.json`` payload for corpus v2 packages.

    Clock rendering fields
    ----------------------
    The Android renderer treats the clock as a **spatial render plane** inserted
    between layer 2 (near midground) and layer 3 (hero foreground).  The
    following fields tell the renderer how to size, place, and occlude the
    clock per-wallpaper:

    * ``clock_plane_index`` — render stack position (3 = after near_mid)
    * ``clock_weight`` — parallax motion weight
    * ``clock_font_scale`` — clock digit size relative to viewport width
    * ``clock_anchor`` — normalised centre (cx, cy) of the clock within
      ``safe_clock_rect``
    * ``safe_clock_rect`` — normalised (l, t, r, b) bounding area where the
      clock can be rendered without heavy subject occlusion
    * ``has_clock_occlusion`` — if True the package ships a
      ``clock_occlusion_mask.webp`` the renderer must apply so the hero
      subject appears *in front of* the clock digits

    All clock position / size values are **normalised** so the renderer can
    scale to any device resolution.
    """

    wallpaper_id: str
    version: int = PARALLAX_FORMAT_VERSION
    target_device: str = TARGET_DEVICE_NAME
    resolution: tuple[int, int] = TARGET_RENDER_RESOLUTION
    layer_count: int = 5
    clock_plane_index: int = 3
    parallax_strength: float = 0.65
    overscan: float = 0.18
    motion_profile: str = "cinematic_slow"
    depth_weights: list[float] = field(
        default_factory=lambda: DEFAULT_DEPTH_WEIGHTS.copy()
    )
    blur_px: list[float] = field(default_factory=lambda: DEFAULT_LAYER_BLUR.copy())

    # ---- Dynamic clock rendering metadata ----
    clock_weight: float = DEFAULT_CLOCK_WEIGHT
    clock_font_scale: float = DEFAULT_CLOCK_FONT_SCALE
    clock_anchor: tuple[float, float] = DEFAULT_CLOCK_ANCHOR
    safe_clock_rect: tuple[float, float, float, float] = (0.16, 0.07, 0.84, 0.30)
    has_clock_occlusion: bool = True

    focus_anchor: tuple[float, float] = (0.50, 0.36)
    subject_bbox: tuple[float, float, float, float] = (0.27, 0.14, 0.73, 0.87)
    inpainted: bool = True
    depth_model: str = "depth_anything_v2_large"
    segmentation_model: str = "birefnet"
    quality: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["id"] = payload.pop("wallpaper_id")
        return payload


@dataclass(slots=True)
class WallpaperPackage:
    wallpaper_id: str
    title: str
    meta: WallpaperMeta
    layers: list[LayerSpec]
    preview_asset: str
    rendered_assets: dict[str, bytes] = field(default_factory=dict, repr=False)

    @property
    def package_dir(self) -> str:
        return self.wallpaper_id

    def required_asset_names(self) -> list[str]:
        return PACKAGE_CONTRACT.required_asset_names(self.layers)

    def validate_rendered_assets(self) -> None:
        expected = set(self.required_asset_names())
        actual = set(self.rendered_assets)
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(
                f"Missing rendered assets for package {self.wallpaper_id}: {missing}"
            )

        empty = sorted(
            asset_name
            for asset_name, payload in self.rendered_assets.items()
            if not payload
        )
        if empty:
            raise ValueError(
                f"Empty rendered assets for package {self.wallpaper_id}: {empty}"
            )


@dataclass(slots=True)
class CorpusIndexEntry:
    wallpaper_id: str
    title: str
    preview: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.wallpaper_id, "title": self.title, "preview": self.preview}


@dataclass(slots=True)
class CorpusManifest:
    wallpapers: list[CorpusIndexEntry]
    format: str = PARALLAX_CORPUS_FORMAT

    def write(self, path: Path) -> None:
        payload = {
            "format": self.format,
            "wallpapers": [entry.to_dict() for entry in self.wallpapers],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
