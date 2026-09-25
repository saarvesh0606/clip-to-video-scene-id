"""Shared plumbing for the torch-based embedders: devices, weight downloads, batching."""

import hashlib
import logging
import os
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def cache_dir() -> Path:
    return Path(os.environ.get("SCENEID_CACHE_DIR", Path.home() / ".cache" / "sceneid"))


def fetch_weights(url: str, sha256: str) -> Path:
    """Download model weights once into the cache and check their sha256."""
    path = cache_dir() / "models" / url.rsplit("/", 1)[-1]
    if path.is_file() and _sha256(path) == sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    log.info("embedder.download", extra={"url": url})
    urllib.request.urlretrieve(url, part)
    digest = _sha256(part)
    if digest != sha256:
        part.unlink()
        raise RuntimeError(f"{url}: sha256 {digest} does not match the pinned {sha256}")
    part.replace(path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def normalize_imagenet(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 3xHxW, ImageNet mean/std normalised."""
    x = frame.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def run_batched(
    frames: Sequence[np.ndarray],
    prepare: Callable[[np.ndarray], np.ndarray],
    forward: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Prepare frames, group equal shapes into batches, run `forward`, restore the order."""
    prepared = [prepare(f) for f in frames]
    by_shape: dict[tuple, list[int]] = defaultdict(list)
    for i, x in enumerate(prepared):
        by_shape[x.shape].append(i)
    out: list[np.ndarray | None] = [None] * len(prepared)
    for indices in by_shape.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            vectors = forward(np.stack([prepared[i] for i in chunk]))
            for i, v in zip(chunk, vectors, strict=True):
                out[i] = v
    return np.stack(out)
