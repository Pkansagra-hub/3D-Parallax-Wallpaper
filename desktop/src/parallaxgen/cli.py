from __future__ import annotations

import json
from pathlib import Path

import typer

from parallaxgen.compose.scene_builder import build_scene_package
from parallaxgen.corpus.manifest import build_manifest, write_package_files
from parallaxgen.corpus.packer import pack_corpus_directory
from parallaxgen.preview.preview_renderer import render_preview_summary

app = typer.Typer(help="ParallaxGen desktop tooling")


@app.command()
def process(
    image: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., file_okay=False, dir_okay=True),
    title: str | None = typer.Option(None, help="Optional display title"),
) -> None:
    """Generate a starter wallpaper package from one image."""
    wallpaper = build_scene_package(image_path=image, title=title)
    output.mkdir(parents=True, exist_ok=True)
    write_package_files(output, wallpaper)
    typer.echo(f"Wrote wallpaper package to {output / wallpaper.wallpaper_id}")


@app.command()
def batch(
    image_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Process every image in a directory using the starter pipeline."""
    output.mkdir(parents=True, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    wallpapers = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in supported:
            continue
        wallpaper = build_scene_package(image_path=image_path)
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
) -> None:
    """Render a textual preview summary for one image."""
    preview_payload = render_preview_summary(image_path=image, include_clock=clock)
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
