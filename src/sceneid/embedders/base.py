from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Turns RGB frames into L2-normalised float32 vectors, so inner product = cosine."""

    name: str
    dim: int

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Return an ``(len(frames), dim)`` float32 array with unit-length rows."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norms, 1e-12), dtype=np.float32)
