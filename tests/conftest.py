"""Shared fixtures: small synthetic videos and a library built with the tiny embedder.

The videos are rendered from a deterministic function of (seed, time), so a query clip
is the same picture as its reference at the same moment, re-encoded, just like a clip
cut from a real video.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sceneid.embedders import TinyImageEmbedder
from sceneid.indexer import index_video
from sceneid.library import Library
from sceneid.matcher import Matcher
from sceneid.matching import V1Voting

SIZE = (160, 96)  # width, height
VIDEO_FPS = 15.0
INDEX_FPS = 2.0
SCENE_S = 6.0  # real footage holds a shot for seconds, not a few frames


def render_frame(seed: int, t: float) -> np.ndarray:
    """A BGR frame. The scene changes every 6 s; inside a scene a bar slides across."""
    w, h = SIZE
    scene = int(t // SCENE_S)
    rng = np.random.default_rng(seed * 10_000 + scene)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = rng.integers(0, 256, 3)
    for _ in range(4):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 30))
        img[y : y + 30, x : x + 40] = rng.integers(0, 256, 3)
    x = int((t % SCENE_S) / SCENE_S * (w - 20))
    img[h // 2 - 10 : h // 2 + 10, x : x + 20] = 255 - img[0, 0]
    return img


def write_video(path: Path, seed: int, start: float, duration: float) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, SIZE)
    for i in range(int(round(duration * VIDEO_FPS))):
        writer.write(render_frame(seed, start + i / VIDEO_FPS))
    writer.release()
    return path


@pytest.fixture(scope="session")
def videos(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("videos")
    return {
        "ref_a": write_video(d / "ref_a.mp4", seed=1, start=0, duration=30),
        "ref_b": write_video(d / "ref_b.mp4", seed=2, start=0, duration=30),
        "clip_a": write_video(d / "clip_a.mp4", seed=1, start=11.0, duration=5),
        "unknown": write_video(d / "unknown.mp4", seed=99, start=0, duration=5),
    }


@pytest.fixture
def embedder() -> TinyImageEmbedder:
    return TinyImageEmbedder(16)


@pytest.fixture
def library(videos, embedder) -> Library:
    lib = Library(embedder.name, embedder.dim)
    for name in ("ref_a", "ref_b"):
        index_video(lib, embedder, videos[name], fps=INDEX_FPS, short_side=None, batch_size=16)
    return lib


@pytest.fixture
def matcher(library, embedder) -> Matcher:
    return Matcher(
        embedder,
        library,
        V1Voting(),
        query_fps=3.0,
        max_frames=40,
        top_k=10,
        short_side=None,
        batch_size=16,
    )
