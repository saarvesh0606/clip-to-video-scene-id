from collections.abc import Sequence

import cv2
import numpy as np


class PerceptualHashEmbedder:
    """The classic 64-bit DCT perceptual hash (pHash): the no-machine-learning baseline.

    Each bit says whether one low-frequency DCT coefficient of a 32x32 grayscale thumbnail
    is above the median. Bits are stored as +-1/8, a unit vector, so the inner product of
    two hashes is ``1 - 2 * hamming / 64``: inner-product search ranks by Hamming distance,
    and identical frames score 1.0 while unrelated ones score about 0.
    """

    name = "phash64"
    dim = 64

    def embed(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            low = cv2.dct(small)[:8, :8].ravel()
            median = np.median(low[1:])  # the DC term would dominate the median
            rows.append(np.where(low > median, 1.0, -1.0))
        return (np.stack(rows) / 8.0).astype(np.float32)
