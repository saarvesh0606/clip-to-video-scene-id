import logging
import time
from collections.abc import Sequence

import numpy as np

from ._torch import fetch_weights, normalize_imagenet, resolve_device, run_batched
from .base import l2_normalize

log = logging.getLogger(__name__)

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_SHA256 = "9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56"


class SSCDEmbedder:
    """SSCD (Pizzi et al., CVPR 2022), a descriptor trained for image copy detection.

    Uses ``sscd_disc_mixup`` (ResNet-50, 512 dims), the model its authors recommend as the
    default. Preprocessing is their "small edge 288" transform: frames keep their aspect
    ratio with the short side at 288 px (which is what our frame sampler already produces),
    then ImageNet normalisation. Code and weights: facebookresearch/sscd-copy-detection (MIT).
    """

    name = "sscd-disc-mixup"
    dim = 512

    def __init__(self, device: str = "auto", batch_size: int = 64):
        import torch

        started = time.perf_counter()
        self.device = resolve_device(device)
        self.batch_size = batch_size
        weights = fetch_weights(SSCD_URL, SSCD_SHA256)
        self._model = torch.jit.load(str(weights), map_location=self.device).eval()
        log.info(
            "embedder.loaded",
            extra={
                "embedder": self.name,
                "device": self.device,
                "load_s": round(time.perf_counter() - started, 2),
            },
        )

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device)
            return self._model(x).float().cpu().numpy()

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(run_batched(frames, normalize_imagenet, self._forward, self.batch_size))
