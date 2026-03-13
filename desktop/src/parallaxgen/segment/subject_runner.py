from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_hf_loading():
    """Suppress accelerate/transformers weight-materialisation tqdm spam."""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import tqdm.auto as _tqdm_auto

    _orig_tqdm = _tqdm_auto.tqdm

    class _SilentTqdm(_orig_tqdm):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw["disable"] = True
            super().__init__(*a, **kw)

    _tqdm_auto.tqdm = _SilentTqdm  # type: ignore[misc]
    _acc_mod = sys.modules.get("accelerate.utils.modeling")
    if _acc_mod and hasattr(_acc_mod, "tqdm"):
        _acc_mod.tqdm = _SilentTqdm  # type: ignore[attr-defined]
    try:
        import transformers.utils.logging as hf_log

        hf_log.set_verbosity_error()
        yield
    finally:
        hf_log.set_verbosity_warning()
        _tqdm_auto.tqdm = _orig_tqdm
        if _acc_mod and hasattr(_acc_mod, "tqdm"):
            _acc_mod.tqdm = _orig_tqdm  # type: ignore[attr-defined]
        os.environ.pop("TQDM_DISABLE", None)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


_SEGMENTATION_REPOS: dict[str, str] = {
    "birefnet": "ZhengPeng7/BiRefNet",
    "birefnet_hr": "ZhengPeng7/BiRefNet-portrait",
    "birefnet_general": "ZhengPeng7/BiRefNet",
    "rmbg": "briaai/RMBG-2.0",
}

# Input resolution per model variant.
_INPUT_RESOLUTION: dict[str, int] = {
    "birefnet": 1024,
    "birefnet_hr": 1536,
    "birefnet_general": 1024,
    "rmbg": 1024,
}


@dataclass(slots=True)
class SubjectMask:
    alpha: np.ndarray  # float32 [0.0, 1.0], full resolution
    bbox: tuple[float, float, float, float]  # normalised (left, top, right, bottom)
    is_landscape: bool = False  # True when no salient subject detected


class SubjectRunner:
    """Salient-object segmentation via BiRefNet or RMBG.

    Produces a high-resolution alpha matte of the hero subject.  The Android
    renderer uses this to know which pixels go IN FRONT of the clock plane.
    """

    def __init__(
        self,
        model_name: str = "birefnet",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: torch.nn.Module | None = None
        self._processor: object | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return

        if self.model_name == "synthetic":
            return  # no model needed

        if self.model_name not in _SEGMENTATION_REPOS:
            raise ValueError(
                f"Unknown segmentation model '{self.model_name}'. "
                f"Supported: {list(_SEGMENTATION_REPOS) + ['synthetic']}"
            )

        from transformers import AutoModelForImageSegmentation

        repo = _SEGMENTATION_REPOS[self.model_name]
        logger.info("Loading %s from %s on %s …", self.model_name, repo, self.device)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        with _quiet_hf_loading():
            self._model = AutoModelForImageSegmentation.from_pretrained(
                repo, trust_remote_code=True, torch_dtype=dtype
            )
        self._model.to(self.device).eval()  # type: ignore[union-attr]

        if self.device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                "GPU VRAM [BiRefNet loaded]: %.2f GB allocated, %.2f GB reserved",
                alloc,
                reserved,
            )

        # BiRefNet uses torchvision transforms, not an HF image processor.
        # RMBG-2.0 also works best with manual transforms at 1024×1024.
        from torchvision import transforms

        res = _INPUT_RESOLUTION.get(self.model_name, 1024)
        self._processor = transforms.Compose(
            [
                transforms.Resize((res, res)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _inference_dtype(self) -> torch.dtype:
        """Return the dtype expected by the loaded segmentation model."""
        if self._model is None:
            return torch.float32

        try:
            return next(self._model.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, image_path: Path, width: int, height: int) -> SubjectMask:
        """Return a subject alpha matte and normalised bounding box.

        Parameters
        ----------
        image_path:
            Source image.
        width, height:
            Target output dimensions for the alpha matte (must match the
            depth map / layer resolution).
        """
        self._load_model()

        if self.model_name == "synthetic":
            return self._infer_synthetic(width, height)

        image = Image.open(image_path).convert("RGB")

        # Preprocess via torchvision transforms → (1, 3, 1024, 1024)
        assert self._processor is not None
        input_tensor = self._processor(image).unsqueeze(0).to(self.device)  # type: ignore[operator]
        input_tensor = input_tensor.to(dtype=self._inference_dtype())

        # Inference
        assert self._model is not None
        with torch.no_grad():
            outputs = self._model(input_tensor)  # type: ignore[operator]

        # Extract the finest prediction mask
        try:
            mask_logits = self._extract_logits(outputs)
        except RuntimeError as e:
            logger.warning("Logit extraction failed (%s) — treating as landscape", e)
            alpha = np.zeros((height, width), dtype=np.float32)
            return SubjectMask(
                alpha=alpha, bbox=(0.0, 0.0, 1.0, 1.0), is_landscape=True
            )

        # Resize to target dimensions
        if mask_logits.dim() == 2:
            mask_logits = mask_logits.unsqueeze(0).unsqueeze(0)
        elif mask_logits.dim() == 3:
            mask_logits = mask_logits.unsqueeze(0)

        mask_logits = mask_logits.float()

        mask = torch.nn.functional.interpolate(
            mask_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        alpha = torch.sigmoid(mask).squeeze().cpu().numpy().astype(np.float32)

        # Landscape detection: if coverage is <2% (empty) or >90% (full-frame
        # collapse), BiRefNet found no distinct subject — treat as landscape.
        coverage = float((alpha > 0.3).mean())
        if coverage < 0.02 or coverage > 0.90:
            logger.info(
                "Landscape detected (coverage=%.1f%%) — zeroing subject mask",
                coverage * 100,
            )
            alpha = np.zeros_like(alpha)
            return SubjectMask(
                alpha=alpha, bbox=(0.0, 0.0, 1.0, 1.0), is_landscape=True
            )

        bbox = self._compute_bbox(alpha)
        return SubjectMask(alpha=alpha, bbox=bbox)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_logits(outputs: object) -> torch.Tensor:
        """Handle different model output formats."""
        # BiRefNet returns a list of side-output tensors (each [B,1,H,W])
        if isinstance(outputs, (list, tuple)):
            last = outputs[-1]
            # Nested list: pick last tensor from the last group
            if isinstance(last, (list, tuple)):
                last = last[-1]
            return last
        # HuggingFace model output with .logits
        if hasattr(outputs, "logits"):
            return outputs.logits  # type: ignore[union-attr]
        raise RuntimeError(f"Cannot extract logits from model output: {type(outputs)}")

    @staticmethod
    def _compute_bbox(
        alpha: np.ndarray, threshold: float = 0.3
    ) -> tuple[float, float, float, float]:
        """Derive a normalised bounding box from the alpha matte."""
        binary = alpha > threshold
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not rows.any() or not cols.any():
            return (0.0, 0.0, 1.0, 1.0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h, w = alpha.shape
        return (float(cmin / w), float(rmin / h), float(cmax / w), float(rmax / h))

    @staticmethod
    def _infer_synthetic(width: int, height: int) -> SubjectMask:
        """Deterministic rectangle for unit tests (no model download)."""
        alpha = np.zeros((height, width), dtype=np.float32)
        left, right = int(width * 0.27), int(width * 0.73)
        top, bottom = int(height * 0.14), int(height * 0.87)
        alpha[top:bottom, left:right] = 1.0
        return SubjectMask(alpha=alpha, bbox=(0.27, 0.14, 0.73, 0.87))
