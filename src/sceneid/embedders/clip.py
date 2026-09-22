import logging
import time
from collections.abc import Sequence

import numpy as np

from .base import l2_normalize

log = logging.getLogger(__name__)


class ClipEmbedder:
    """OpenAI CLIP image encoder via sentence-transformers."""

    def __init__(self, model_name: str, name: str, device: str = "auto", batch_size: int = 32):
        import torch  # imported lazily: torch is an optional dependency
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        self._model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.batch_size = batch_size
        self.name = name
        probe = self._model.encode([_blank()], convert_to_numpy=True)
        self.dim = int(probe.shape[1])
        log.info(
            "embedder.loaded",
            extra={
                "embedder": name,
                "device": device,
                "dim": self.dim,
                "load_s": round(time.perf_counter() - started, 2),
            },
        )

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image

        if not frames:
            return np.zeros((0, self.dim), dtype=np.float32)
        images = [Image.fromarray(f) for f in frames]
        vectors = self._model.encode(
            images,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)


def _blank():
    from PIL import Image

    return Image.new("RGB", (224, 224))
