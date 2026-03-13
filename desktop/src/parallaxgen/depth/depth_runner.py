from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DEPTH_ANYTHING_REPOS: dict[str, str] = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}


@dataclass(slots=True)
class DepthResult:
    width: int
    height: int
    depth_map: np.ndarray  # float32 [0.0, 1.0], 0=near 1=far
    model_name: str


class DepthRunner:
    """Runs monocular depth estimation on a single image.

    Supported models:
    - ``depth_anything_v2_small`` / ``_base`` / ``_large`` via HuggingFace
      ``transformers`` (default: large).
    - ``midas_dpt_large`` via ``torch.hub`` (requires optional ``timm``
      dependency).
    """

    def __init__(
        self,
        model_name: str = "depth_anything_v2_large",
        output_resolution: tuple[int, int] = (1440, 3120),
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.output_resolution = output_resolution
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: torch.nn.Module | None = None
        self._processor: object | None = None
        self._backend: str = ""

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return

        if self.model_name in _DEPTH_ANYTHING_REPOS:
            self._load_depth_anything()
        elif self.model_name == "midas_dpt_large":
            self._load_midas()
        elif self.model_name == "depth_pro":
            self._load_depth_pro()
        elif self.model_name == "synthetic":
            self._backend = "synthetic"
        else:
            raise ValueError(
                f"Unknown depth model '{self.model_name}'. "
                f"Supported: {list(_DEPTH_ANYTHING_REPOS) + ['midas_dpt_large', 'depth_pro', 'synthetic']}"
            )

    def _load_depth_anything(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = _DEPTH_ANYTHING_REPOS[self.model_name]
        logger.info("Loading %s from %s on %s …", self.model_name, repo, self.device)
        self._processor = AutoImageProcessor.from_pretrained(repo)
        self._model = AutoModelForDepthEstimation.from_pretrained(repo)
        self._model.to(self.device).eval()  # type: ignore[union-attr]
        self._backend = "depth_anything"

    def _load_midas(self) -> None:
        logger.info("Loading MiDaS DPT-Large from torch hub on %s …", self.device)
        self._model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        self._model.to(self.device).eval()  # type: ignore[union-attr]
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self._processor = midas_transforms.dpt_transform
        self._backend = "midas"

    def _load_depth_pro(self) -> None:
        from transformers import AutoImageProcessor, DepthProForDepthEstimation

        repo = "apple/DepthPro-hf"
        logger.info("Loading Depth Pro from %s on %s …", repo, self.device)
        self._processor = AutoImageProcessor.from_pretrained(repo)
        self._model = DepthProForDepthEstimation.from_pretrained(
            repo,
            use_fov_model=False,
        )
        self._model.to(self.device).eval()  # type: ignore[union-attr]
        self._backend = "depth_pro"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, image_path: Path) -> DepthResult:
        """Run depth inference on *image_path* and return a normalised depth map.

        The returned ``depth_map`` is float32 in ``[0, 1]`` where **0 = near**
        and **1 = far**.  The map is resized to ``output_resolution``
        ``(width, height)``.
        """
        self._load_model()

        image = Image.open(image_path).convert("RGB")
        width, height = self.output_resolution

        if self._backend == "depth_anything":
            raw_depth = self._infer_depth_anything(image, height, width)
        elif self._backend == "midas":
            raw_depth = self._infer_midas(image, height, width)
        elif self._backend == "depth_pro":
            raw_depth = self._infer_depth_pro(image, height, width)
        elif self._backend == "synthetic":
            raw_depth = self._infer_synthetic(height, width)
        else:
            raise RuntimeError("Model not loaded")

        # Normalise to [0, 1]
        d_min, d_max = float(raw_depth.min()), float(raw_depth.max())
        if d_max - d_min < 1e-8:
            depth = np.zeros((height, width), dtype=np.float32)
        else:
            depth = ((raw_depth - d_min) / (d_max - d_min)).astype(np.float32)

        return DepthResult(
            width=width,
            height=height,
            depth_map=depth,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Backend-specific inference
    # ------------------------------------------------------------------
    def _infer_depth_anything(
        self, image: Image.Image, height: int, width: int
    ) -> np.ndarray:
        assert self._model is not None and self._processor is not None
        inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[operator]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predicted_depth = self._model(**inputs).predicted_depth  # type: ignore[operator]

        # Resize to target and move to numpy
        depth = (
            torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        # Depth Anything V2: larger values = farther (matches our convention)
        return depth

    def _infer_midas(self, image: Image.Image, height: int, width: int) -> np.ndarray:
        assert self._model is not None and self._processor is not None
        img_np = np.array(image)
        input_batch = self._processor(img_np).to(self.device)  # type: ignore[operator]

        with torch.no_grad():
            prediction = self._model(input_batch)  # type: ignore[operator]

        depth = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        # MiDaS: higher value = nearer.  Invert to match our convention (0=near 1=far).
        return depth.max() - depth

    def _infer_depth_pro(
        self, image: Image.Image, height: int, width: int
    ) -> np.ndarray:
        """Depth Pro produces metric depth — larger values are farther."""
        assert self._model is not None and self._processor is not None
        inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[operator]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[operator]

        # Use post_process_depth_estimation for proper metric depth.
        post = self._processor.post_process_depth_estimation(  # type: ignore[union-attr]
            outputs,
            target_sizes=[(height, width)],
        )
        depth = post[0]["predicted_depth"].cpu().numpy()
        # Depth Pro: larger values = farther (matches our 0=near 1=far convention)
        return depth

    @staticmethod
    def _infer_synthetic(height: int, width: int) -> np.ndarray:
        """Deterministic vertical ramp for unit tests and CI (no model download)."""
        return np.tile(
            np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], (1, width)
        )
