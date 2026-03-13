from __future__ import annotations

import zipfile
from pathlib import Path


def pack_corpus_directory(corpus_dir: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in corpus_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(corpus_dir))
    return output_path
