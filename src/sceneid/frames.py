"""Reading videos and sampling frames at a fixed rate.

Timestamps come from each frame's presentation time (``CAP_PROP_POS_MSEC``), not from
``frame_index / average_fps``: phone and screen recordings are variable-frame-rate, and
the index-based estimate drifts by hundreds of milliseconds on them.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import cv2
import numpy as np


class VideoError(Exception):
    """The file is missing, unreadable, or not a video OpenCV can decode."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_s: float
    avg_fps: float
    frame_count: int
    width: int
    height: int


def _open(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise VideoError(f"video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"could not open video: {path.name}")
    return cap


def probe(path: str | Path) -> VideoInfo:
    """Read container metadata. The duration is approximate for variable-frame-rate files."""
    path = Path(path)
    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        raise VideoError(f"not a decodable video: {path.name}")
    return VideoInfo(path, count / fps, fps, count, width, height)


def resize_short_side(frame: np.ndarray, short_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = short_side / min(h, w)
    if scale >= 1.0:
        return frame
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def sample_frames(
    path: str | Path,
    fps: float,
    *,
    short_side: int | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_s, rgb_frame)`` at roughly ``fps`` samples per second.

    Samples sit on a regular grid (0, 1/fps, 2/fps, ...): each grid point takes the first
    decoded frame at or after it. Frames are streamed, so memory stays flat for long videos.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    path = Path(path)
    cap = _open(path)
    avg_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = 1.0 / fps
    next_t = 0.0
    prev_t = -1.0
    emitted = 0
    try:
        while max_frames is None or emitted < max_frames:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= prev_t:
                # Some containers report no timestamps; fall back to a frame-rate estimate.
                t = prev_t + 1.0 / avg_fps
            prev_t = t
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            while next_t <= t + 1e-6:
                next_t += interval
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if short_side:
                frame = resize_short_side(frame, short_side)
            yield t, frame
            emitted += 1
    finally:
        cap.release()


def batched(items: Iterable, n: int) -> Iterator[list]:
    it = iter(items)
    while batch := list(islice(it, n)):
        yield batch
