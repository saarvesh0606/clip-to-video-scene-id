import json

import numpy as np
import pytest

from sceneid.embedders import l2_normalize
from sceneid.library import Library, LibraryError

DIM = 8


def _vectors(n: int, seed: int) -> np.ndarray:
    return l2_normalize(np.random.default_rng(seed).normal(size=(n, DIM)))


def _add(lib: Library, video_id: str, n: int, seed: int) -> np.ndarray:
    vecs = _vectors(n, seed)
    lib.add_video(
        video_id,
        vecs,
        np.arange(n, dtype=np.float32) * 0.5,
        source_path=f"/videos/{video_id}.mp4",
        sha256=f"sha-{video_id}",
        duration_s=n * 0.5,
        sample_fps=2.0,
    )
    return vecs


def test_search_returns_video_and_time_of_the_nearest_frame():
    lib = Library("test", DIM)
    a = _add(lib, "a", 10, seed=1)
    _add(lib, "b", 10, seed=2)
    hits = lib.search(a[[3, 7]], k=1)
    assert hits.video_keys[:, 0].tolist() == [lib.get("a").key] * 2
    assert hits.ref_times[:, 0].tolist() == [1.5, 3.5]
    assert hits.scores[:, 0] == pytest.approx([1.0, 1.0], abs=1e-5)


def test_remove_keeps_the_remaining_rows_aligned():
    lib = Library("test", DIM)
    _add(lib, "a", 10, seed=1)
    b = _add(lib, "b", 12, seed=2)
    lib.remove_video("a")
    assert lib.n_videos == 1 and lib.n_vectors == 12
    hits = lib.search(b[[0, 11]], k=3)
    assert set(hits.video_keys.ravel().tolist()) == {lib.get("b").key}
    assert hits.ref_times[:, 0].tolist() == [0.0, 5.5]


def test_save_and_load_round_trip(tmp_path):
    lib = Library("test", DIM)
    a = _add(lib, "a", 10, seed=1)
    _add(lib, "b", 5, seed=2)
    lib.remove_video("b")
    _add(lib, "c", 7, seed=3)
    lib.save(tmp_path)

    loaded = Library.load(tmp_path)
    assert [v.video_id for v in loaded.videos()] == ["a", "c"]
    assert loaded.n_vectors == 17
    hits = loaded.search(a[[4]], k=1)
    assert loaded.video_id(int(hits.video_keys[0, 0])) == "a"
    assert hits.ref_times[0, 0] == 2.0
    # New ids keep counting up after a reload, so they never collide.
    _add(loaded, "d", 3, seed=4)
    assert loaded.n_vectors == 20


def test_fewer_vectors_than_k_gives_empty_slots():
    lib = Library("test", DIM)
    a = _add(lib, "a", 3, seed=1)
    hits = lib.search(a[:1], k=5)
    assert hits.video_keys[0].tolist()[3:] == [-1, -1]


@pytest.mark.parametrize("bad_id", ["", "has space", "../escape", "-leading", "x" * 129])
def test_invalid_video_ids_are_rejected(bad_id):
    with pytest.raises(LibraryError, match="invalid video id"):
        _add(Library("test", DIM), bad_id, 2, seed=0)


def test_duplicate_id_and_wrong_shapes_are_rejected():
    lib = Library("test", DIM)
    _add(lib, "a", 2, seed=0)
    with pytest.raises(LibraryError, match="already"):
        _add(lib, "a", 2, seed=1)
    with pytest.raises(LibraryError, match="shape"):
        lib.search(np.zeros((1, DIM + 1), np.float32), k=1)
    with pytest.raises(LibraryError, match="empty"):
        Library("test", DIM).search(np.zeros((1, DIM), np.float32), k=1)


def test_embedder_mismatch_is_refused(tmp_path):
    lib = Library("clip-vit-b32", DIM)
    _add(lib, "a", 2, seed=0)
    lib.save(tmp_path)
    with pytest.raises(LibraryError, match="built with clip-vit-b32"):
        Library.open_or_create(tmp_path, "tiny16", 256)


def test_tampered_library_fails_to_load(tmp_path):
    lib = Library("test", DIM)
    _add(lib, "a", 4, seed=0)
    lib.save(tmp_path)
    videos = json.loads((tmp_path / "videos.json").read_text())
    videos[0]["n_vectors"] = 99
    (tmp_path / "videos.json").write_text(json.dumps(videos))
    with pytest.raises(LibraryError, match="inconsistent"):
        Library.load(tmp_path)
