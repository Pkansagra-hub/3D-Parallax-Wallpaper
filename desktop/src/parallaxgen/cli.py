from __future__ import annotations

import json
from pathlib import Path

import typer

from parallaxgen.compose.scene_builder import build_scene_package
from parallaxgen.config import PipelineConfig, build_pipeline_config
from parallaxgen.corpus.manifest import build_manifest, write_package_files
from parallaxgen.corpus.packer import pack_corpus_directory
from parallaxgen.preview.preview_renderer import render_preview_summary

app = typer.Typer(help="ParallaxGen desktop tooling")


def _resolve_config(
    *,
    width: int,
    height: int,
    depth_model: str,
    segmentation_model: str,
    overscan: float,
    parallax_strength: float,
    motion_profile: str,
    clock_safe_rect: str | None,
    min_depth_separation: float,
    min_subject_coverage: float,
    min_clock_clearance: float,
) -> PipelineConfig:
    return build_pipeline_config(
        width=width,
        height=height,
        depth_model=depth_model,
        segmentation_model=segmentation_model,
        overscan=overscan,
        parallax_strength=parallax_strength,
        motion_profile=motion_profile,
        clock_safe_rect=clock_safe_rect,
        min_depth_separation=min_depth_separation,
        min_subject_coverage=min_subject_coverage,
        min_clock_clearance=min_clock_clearance,
    )


def _emit_config_debug(
    config: PipelineConfig, print_config: bool, config_out: Path | None
) -> None:
    if print_config:
        typer.echo(config.to_json())
    if config_out is not None:
        config_out.parent.mkdir(parents=True, exist_ok=True)
        config.write_json(config_out)


@app.command()
def process(
    image: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., file_okay=False, dir_okay=True),
    title: str | None = typer.Option(None, help="Optional display title"),
    width: int = typer.Option(1440, help="Output width for exported assets."),
    height: int = typer.Option(3120, help="Output height for exported assets."),
    depth_model: str = typer.Option("midas_dpt_large", help="Depth model name."),
    segmentation_model: str = typer.Option(
        "birefnet_or_equivalent", help="Segmentation model name."
    ),
    overscan: float = typer.Option(0.18, help="Overscan ratio."),
    parallax_strength: float = typer.Option(0.65, help="Parallax strength."),
    motion_profile: str = typer.Option("cinematic_slow", help="Motion profile name."),
    clock_safe_rect: str | None = typer.Option(
        None,
        help="Clock safe rect as left,top,right,bottom normalized floats.",
    ),
    min_depth_separation: float = typer.Option(
        0.18, help="Minimum depth separation threshold."
    ),
    min_subject_coverage: float = typer.Option(
        0.08, help="Minimum subject coverage threshold."
    ),
    min_clock_clearance: float = typer.Option(
        0.55, help="Minimum clock clearance threshold."
    ),
    print_config: bool = typer.Option(
        False, help="Print the resolved pipeline config."
    ),
    config_out: Path | None = typer.Option(
        None, help="Optional path to write resolved config JSON."
    ),
) -> None:
    """Generate a starter wallpaper package from one image."""
    config = _resolve_config(
        width=width,
        height=height,
        depth_model=depth_model,
        segmentation_model=segmentation_model,
        overscan=overscan,
        parallax_strength=parallax_strength,
        motion_profile=motion_profile,
        clock_safe_rect=clock_safe_rect,
        min_depth_separation=min_depth_separation,
        min_subject_coverage=min_subject_coverage,
        min_clock_clearance=min_clock_clearance,
    )
    _emit_config_debug(config, print_config=print_config, config_out=config_out)

    wallpaper = build_scene_package(image_path=image, title=title, config=config)
    output.mkdir(parents=True, exist_ok=True)
    write_package_files(output, wallpaper)
    typer.echo(f"Wrote wallpaper package to {output / wallpaper.wallpaper_id}")


@app.command()
def batch(
    image_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output: Path = typer.Option(..., file_okay=False, dir_okay=True),
    width: int = typer.Option(1440, help="Output width for exported assets."),
    height: int = typer.Option(3120, help="Output height for exported assets."),
    depth_model: str = typer.Option("midas_dpt_large", help="Depth model name."),
    segmentation_model: str = typer.Option(
        "birefnet_or_equivalent", help="Segmentation model name."
    ),
    overscan: float = typer.Option(0.18, help="Overscan ratio."),
    parallax_strength: float = typer.Option(0.65, help="Parallax strength."),
    motion_profile: str = typer.Option("cinematic_slow", help="Motion profile name."),
    clock_safe_rect: str | None = typer.Option(
        None,
        help="Clock safe rect as left,top,right,bottom normalized floats.",
    ),
    min_depth_separation: float = typer.Option(
        0.18, help="Minimum depth separation threshold."
    ),
    min_subject_coverage: float = typer.Option(
        0.08, help="Minimum subject coverage threshold."
    ),
    min_clock_clearance: float = typer.Option(
        0.55, help="Minimum clock clearance threshold."
    ),
    print_config: bool = typer.Option(
        False, help="Print the resolved pipeline config."
    ),
    config_out: Path | None = typer.Option(
        None, help="Optional path to write resolved config JSON."
    ),
) -> None:
    """Process every image in a directory using the starter pipeline."""
    config = _resolve_config(
        width=width,
        height=height,
        depth_model=depth_model,
        segmentation_model=segmentation_model,
        overscan=overscan,
        parallax_strength=parallax_strength,
        motion_profile=motion_profile,
        clock_safe_rect=clock_safe_rect,
        min_depth_separation=min_depth_separation,
        min_subject_coverage=min_subject_coverage,
        min_clock_clearance=min_clock_clearance,
    )
    _emit_config_debug(config, print_config=print_config, config_out=config_out)

    output.mkdir(parents=True, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    wallpapers = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in supported:
            continue
        wallpaper = build_scene_package(image_path=image_path, config=config)
        write_package_files(output, wallpaper)
        wallpapers.append(wallpaper)

    build_manifest(wallpapers).write(output / "index.json")
    typer.echo(f"Processed {len(wallpapers)} image(s) into {output}")


@app.command()
def preview(
    image: Path = typer.Argument(..., exists=True, readable=True),
    clock: bool = typer.Option(
        False, help="Include clock placement in preview summary"
    ),
    width: int = typer.Option(1440, help="Output width for exported assets."),
    height: int = typer.Option(3120, help="Output height for exported assets."),
    depth_model: str = typer.Option("midas_dpt_large", help="Depth model name."),
    segmentation_model: str = typer.Option(
        "birefnet_or_equivalent", help="Segmentation model name."
    ),
    overscan: float = typer.Option(0.18, help="Overscan ratio."),
    parallax_strength: float = typer.Option(0.65, help="Parallax strength."),
    motion_profile: str = typer.Option("cinematic_slow", help="Motion profile name."),
    clock_safe_rect: str | None = typer.Option(
        None,
        help="Clock safe rect as left,top,right,bottom normalized floats.",
    ),
    min_depth_separation: float = typer.Option(
        0.18, help="Minimum depth separation threshold."
    ),
    min_subject_coverage: float = typer.Option(
        0.08, help="Minimum subject coverage threshold."
    ),
    min_clock_clearance: float = typer.Option(
        0.55, help="Minimum clock clearance threshold."
    ),
) -> None:
    """Render a textual preview summary for one image."""
    config = _resolve_config(
        width=width,
        height=height,
        depth_model=depth_model,
        segmentation_model=segmentation_model,
        overscan=overscan,
        parallax_strength=parallax_strength,
        motion_profile=motion_profile,
        clock_safe_rect=clock_safe_rect,
        min_depth_separation=min_depth_separation,
        min_subject_coverage=min_subject_coverage,
        min_clock_clearance=min_clock_clearance,
    )
    preview_payload = render_preview_summary(
        image_path=image,
        include_clock=clock,
        config=config,
    )
    typer.echo(json.dumps(preview_payload, indent=2))


@app.command()
def pack(
    corpus_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    out: Path = typer.Option(..., help="Path to the output zip archive"),
) -> None:
    """Zip a prepared corpus directory into a .parallax-style archive."""
    archive_path = pack_corpus_directory(corpus_dir=corpus_dir, output_path=out)
    typer.echo(f"Packed archive at {archive_path}")


@app.command()
def inspect(package_dir: Path = typer.Argument(..., exists=True)) -> None:
    """Print the meta.json from a generated wallpaper directory."""
    meta_path = package_dir / "meta.json" if package_dir.is_dir() else package_dir
    typer.echo(meta_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
