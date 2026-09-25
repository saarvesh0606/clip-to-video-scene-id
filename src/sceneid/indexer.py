"""Adding reference videos to a library."""

import hashlib
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedders import Embedder
from .frames import batched, probe, sample_frames
from .library import Library, LibraryError, VideoRecord

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStats:
    record: VideoRecord
    decode_s: float
    embed_s: float

    @property
    def realtime_factor(self) -> float:
        """Seconds of video indexed per second of wall time."""
        return self.record.duration_s / max(1e-9, self.decode_s + self.embed_s)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def default_video_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-.")[:128] or "video"


@dataclass(frozen=True)
class VideoEmbeddings:
    times: np.ndarray  # (n,) seconds
    vectors: dict[str, np.ndarray]  # embedder name -> (n, dim)
    decode_s: float
    embed_s: dict[str, float]  # embedder name -> seconds


def embed_video_multi(
    embedders: Sequence[Embedder],
    path: Path,
    *,
    fps: float,
    short_side: int | None,
    batch_size: int,
    max_frames: int | None = None,
) -> VideoEmbeddings:
    """Decode a video once and embed its sampled frames with every embedder, in batches."""
    times: list[float] = []
    vectors: dict[str, list[np.ndarray]] = {e.name: [] for e in embedders}
    embed_s = dict.fromkeys(vectors, 0.0)
    decode_s = 0.0
    frames = sample_frames(path, fps, short_side=short_side, max_frames=max_frames)
    try:
        while True:
            started = time.perf_counter()
            batch = next(batched(frames, batch_size), None)
            decode_s += time.perf_counter() - started
            if not batch:
                break
            images = [frame for _, frame in batch]
            for e in embedders:
                started = time.perf_counter()
                vectors[e.name].append(e.embed(images))
                embed_s[e.name] += time.perf_counter() - started
            times.extend(t for t, _ in batch)
    finally:
        frames.close()  # releases the decoder even if embedding raised
    stacked = {
        e.name: np.concatenate(vectors[e.name]) if times else np.zeros((0, e.dim), np.float32)
        for e in embedders
    }
    return VideoEmbeddings(np.array(times, dtype=np.float32), stacked, decode_s, embed_s)


def embed_video(
    embedder: Embedder,
    path: Path,
    *,
    fps: float,
    short_side: int | None,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sample and embed a video in batches. Returns (times, vectors, decode_s, embed_s)."""
    out = embed_video_multi(
        [embedder],
        path,
        fps=fps,
        short_side=short_side,
        batch_size=batch_size,
        max_frames=max_frames,
    )
    return out.times, out.vectors[embedder.name], out.decode_s, out.embed_s[embedder.name]


def index_video(
    library: Library,
    embedder: Embedder,
    path: str | Path,
    *,
    video_id: str | None = None,
    fps: float,
    short_side: int | None,
    batch_size: int,
    replace: bool = False,
) -> IndexStats:
    path = Path(path).resolve()
    video_id = video_id or default_video_id(path)
    info = probe(path)
    sha = file_sha256(path)

    existing = library.get(video_id)
    if existing and not replace:
        raise LibraryError(f"{video_id!r} is already indexed; pass --replace to re-index it")
    duplicate = library.find_by_sha256(sha)
    if duplicate and duplicate.video_id != video_id:
        raise LibraryError(f"{path.name} is already indexed as {duplicate.video_id!r}")

    times, vectors, decode_s, embed_s = embed_video(
        embedder, path, fps=fps, short_side=short_side, batch_size=batch_size
    )
    # Only touch the library once embedding has succeeded.
    if existing:
        library.remove_video(video_id)
    record = library.add_video(
        video_id,
        vectors,
        times,
        source_path=str(path),
        sha256=sha,
        duration_s=info.duration_s,
        sample_fps=fps,
    )
    stats = IndexStats(record, decode_s, embed_s)
    log.info(
        "index.video_done",
        extra={
            "video_id": video_id,
            "duration_s": record.duration_s,
            "frames": record.n_vectors,
            "decode_s": round(decode_s, 2),
            "embed_s": round(embed_s, 2),
            "realtime_x": round(stats.realtime_factor, 1),
            "replaced": bool(existing),
        },
    )
    return stats
