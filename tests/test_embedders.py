import cv2
import numpy as np
import pytest

from sceneid.embedders import EMBEDDERS, PerceptualHashEmbedder, create_embedder
from tests.conftest import render_frame


def _rgb(seed: int, t: float) -> np.ndarray:
    return cv2.cvtColor(render_frame(seed, t), cv2.COLOR_BGR2RGB)


def test_phash_scores_are_one_minus_twice_the_hamming_fraction():
    e = PerceptualHashEmbedder()
    a, b = _rgb(1, 1.0), _rgb(2, 1.0)
    va, vb = e.embed([a, b])
    assert np.linalg.norm(va) == pytest.approx(1.0)
    bits_a, bits_b = va > 0, vb > 0
    hamming = int((bits_a != bits_b).sum())
    assert float(va @ vb) == pytest.approx(1 - 2 * hamming / 64)
    assert float(va @ va) == pytest.approx(1.0)


def test_phash_survives_compression_but_not_mirroring():
    e = PerceptualHashEmbedder()
    frame = cv2.resize(_rgb(1, 1.0), (640, 384), interpolation=cv2.INTER_CUBIC)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
    compressed = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    v = e.embed([frame, compressed, frame[:, ::-1].copy()])
    assert v[0] @ v[1] > 0.9
    assert v[0] @ v[2] < 0.7


def test_registry_knows_every_embedder():
    assert {"clip-vit-b32", "phash64", "tiny16"} <= set(EMBEDDERS)
    with pytest.raises(ValueError, match="unknown embedder"):
        create_embedder("resnet-9000")
