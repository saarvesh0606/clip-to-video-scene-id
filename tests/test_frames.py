import numpy as np
import pytest

from sceneid.frames import VideoError, batched, probe, resize_short_side, sample_frames
from tests.conftest import VIDEO_FPS


def test_probe_reads_metadata(videos):
    info = probe(videos["ref_a"])
    assert info.width == 160 and info.height == 96
    assert info.avg_fps == pytest.approx(VIDEO_FPS)
    assert info.duration_s == pytest.approx(30.0, abs=0.1)


def test_samples_sit_on_the_requested_grid(videos):
    times = np.array([t for t, _ in sample_frames(videos["ref_a"], fps=2.0)])
    assert len(times) == 60
    grid = np.arange(60) * 0.5
    # Each sample is the first frame at or after its grid point.
    assert np.all(times >= grid - 1e-6)
    assert np.all(times - grid < 1 / VIDEO_FPS + 1e-6)


def test_frames_are_rgb(videos):
    _, frame = next(sample_frames(videos["ref_a"], fps=1.0))
    assert frame.shape == (96, 160, 3) and frame.dtype == np.uint8


def test_max_frames_is_respected(videos):
    assert len(list(sample_frames(videos["ref_a"], fps=3.0, max_frames=7))) == 7


def test_short_side_resize_keeps_aspect_and_never_upscales():
    frame = np.zeros((96, 160, 3), np.uint8)
    assert resize_short_side(frame, 48).shape == (48, 80, 3)
    assert resize_short_side(frame, 500).shape == frame.shape


def test_missing_and_garbage_files_raise_video_error(tmp_path):
    with pytest.raises(VideoError):
        probe(tmp_path / "nope.mp4")
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(np.random.default_rng(0).bytes(4096))
    with pytest.raises(VideoError):
        probe(garbage)


def test_batched():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(batched([], 3)) == []
