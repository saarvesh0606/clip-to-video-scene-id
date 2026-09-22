from collections.abc import Sequence

import cv2
import numpy as np

from .base import l2_normalize


class TinyImageEmbedder:
    """A 16x16 grayscale thumbnail, mean-centred. No model, no training.

    It is the "no machine learning" baseline in benchmarks, and it keeps the test suite
    fast and free of torch.
    """

    def __init__(self, size: int = 16):
        self.size = size
        self.dim = size * size
        self.name = f"tiny{size}"

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            thumb = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
            v = thumb.astype(np.float32).ravel()
            rows.append(v - v.mean())
        return l2_normalize(np.stack(rows))
