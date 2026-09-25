import cv2
import numpy as np
import pytest

from sceneid.embedders import EMBEDDERS, PerceptualHashEmbedder, create_embedder
from sceneid.embedders._torch import run_batched
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


def test_batching_groups_shapes_and_keeps_the_original_order():
    frames = [np.full((h, w, 3), i, np.uint8) for i, (h, w) in enumerate([(4, 6), (5, 5), (4, 6)])]
    calls = []

    def forward(batch):
        calls.append(batch.shape)
        return batch.reshape(len(batch), -1)[:, :1].astype(np.float32)

    out = run_batched(frames, lambda f: f.astype(np.float32), forward, batch_size=8)
    assert out[:, 0].tolist() == [0.0, 1.0, 2.0]  # same order as the input
    assert sorted(s[0] for s in calls) == [1, 2]  # one batch per shape


def test_registry_knows_every_embedder():
    assert {"clip-vit-b32", "sscd-disc-mixup", "dinov2-base", "phash64", "tiny16"} <= set(EMBEDDERS)
    with pytest.raises(ValueError, match="unknown embedder"):
        create_embedder("resnet-9000")


@pytest.mark.parametrize("name", ["sscd-disc-mixup", "dinov2-base"])
def test_neural_embedders_give_unit_vectors_independent_of_batching(name):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        e = create_embedder(name, device="cpu", batch_size=2)
    except OSError as exc:  # weights not downloadable here (e.g. offline CI)
        pytest.skip(f"weights unavailable: {exc}")
    wide = cv2.resize(_rgb(1, 1.0), (512, 288))
    square = cv2.resize(_rgb(2, 1.0), (288, 288))
    together = e.embed([wide, square, wide])
    assert together.shape == (3, e.dim)
    assert np.allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-4)
    alone = e.embed([square])
    assert np.allclose(together[1], alone[0], atol=1e-4)
