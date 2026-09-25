import logging
import time
from collections.abc import Sequence

import cv2
import numpy as np

from ._torch import normalize_imagenet, resolve_device, run_batched
from .base import l2_normalize

log = logging.getLogger(__name__)


class DinoV2Embedder:
    """DINOv2 (Oquab et al., 2023), self-supervised features that are strong at matching
    specific images. Uses ``facebook/dinov2-base`` (ViT-B/14, 768 dims, Apache-2.0) via
    transformers, taking the normalised CLS token.

    Frames are resized to 224x224 without cropping, so the whole frame is seen; the
    standard "resize then centre-crop" transform would cut the sides off widescreen frames.
    """

    name = "dinov2-base"
    dim = 768
    model_id = "facebook/dinov2-base"

    def __init__(self, device: str = "auto", batch_size: int = 64):
        import torch
        from transformers import AutoModel

        started = time.perf_counter()
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self._torch = torch
        log.info(
            "embedder.loaded",
            extra={
                "embedder": self.name,
                "device": self.device,
                "load_s": round(time.perf_counter() - started, 2),
            },
        )

    @staticmethod
    def _prepare(frame: np.ndarray) -> np.ndarray:
        return normalize_imagenet(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            x = self._torch.from_numpy(batch).to(self.device)
            return self._model(pixel_values=x).pooler_output.float().cpu().numpy()

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(run_batched(frames, self._prepare, self._forward, self.batch_size))
