"""The reference library: one vector per sampled frame, plus which video and time it came from.

On disk a library is a directory:

    manifest.json   schema version, embedder name and dimension, id counters
    vectors.faiss   the FAISS index (exact inner-product search over unit vectors = cosine)
    rows.npz        for every vector id: the video it belongs to and its timestamp
    videos.json     one record per indexed video

The embedder is recorded so a library built with one model can never be queried with another.
Each file is replaced atomically, and loading cross-checks all four, so a half-written or
hand-edited library fails loudly instead of returning wrong matches.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

SCHEMA_VERSION = 1
_MANIFEST = "manifest.json"
_INDEX = "vectors.faiss"
_ROWS = "rows.npz"
_VIDEOS = "videos.json"

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LibraryError(Exception):
    pass


@dataclass
class VideoRecord:
    video_id: str
    key: int
    source_path: str
    sha256: str
    duration_s: float
    sample_fps: float
    n_vectors: int
    indexed_at: str


@dataclass(frozen=True)
class SearchHits:
    """Nearest reference frames for each query frame, best first."""

    scores: np.ndarray  # (n_query, k) float32 cosine similarity
    video_keys: np.ndarray  # (n_query, k) int32 video key, -1 where there was no result
    ref_times: np.ndarray  # (n_query, k) float32 seconds into that reference video


class Library:
    def __init__(self, embedder: str, dim: int):
        self.embedder = embedder
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._ids = np.zeros(0, dtype=np.int64)  # sorted: ids are handed out in increasing order
        self._keys = np.zeros(0, dtype=np.int32)
        self._times = np.zeros(0, dtype=np.float32)
        self._videos: dict[int, VideoRecord] = {}
        self._next_id = 0
        self._next_key = 0

    # ------------------------------------------------------------------ queries

    @property
    def n_videos(self) -> int:
        return len(self._videos)

    @property
    def n_vectors(self) -> int:
        return int(self._index.ntotal)

    @property
    def index_bytes(self) -> int:
        return self.n_vectors * self.dim * 4

    def videos(self) -> list[VideoRecord]:
        return sorted(self._videos.values(), key=lambda v: v.video_id)

    def get(self, video_id: str) -> VideoRecord | None:
        return next((v for v in self._videos.values() if v.video_id == video_id), None)

    def find_by_sha256(self, sha256: str) -> VideoRecord | None:
        return next((v for v in self._videos.values() if v.sha256 == sha256), None)

    def video_id(self, key: int) -> str:
        return self._videos[key].video_id

    def search(self, queries: np.ndarray, k: int) -> SearchHits:
        if self.n_vectors == 0:
            raise LibraryError("the library is empty; index some videos first")
        queries = self._check_vectors(queries)
        scores, ids = self._index.search(queries, k)
        valid = ids >= 0
        rows = np.clip(np.searchsorted(self._ids, ids), 0, len(self._ids) - 1)
        keys = np.where(valid, self._keys[rows], -1).astype(np.int32)
        times = np.where(valid, self._times[rows], 0.0).astype(np.float32)
        return SearchHits(scores.astype(np.float32), keys, times)

    # ---------------------------------------------------------------- mutation

    def add_video(
        self,
        video_id: str,
        embeddings: np.ndarray,
        times: np.ndarray,
        *,
        source_path: str,
        sha256: str,
        duration_s: float,
        sample_fps: float,
    ) -> VideoRecord:
        if not VIDEO_ID_RE.match(video_id):
            raise LibraryError(
                f"invalid video id {video_id!r}: use letters, digits, '.', '_' or '-' (max 128)"
            )
        if self.get(video_id):
            raise LibraryError(f"video id {video_id!r} is already in the library")
        embeddings = self._check_vectors(embeddings)
        times = np.asarray(times, dtype=np.float32)
        if len(embeddings) == 0:
            raise LibraryError(f"no frames were sampled from {video_id!r}")
        if times.shape != (len(embeddings),):
            raise LibraryError("need exactly one timestamp per embedding")

        key = self._next_key
        ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)
        self._ids = np.concatenate([self._ids, ids])
        self._keys = np.concatenate([self._keys, np.full(len(ids), key, dtype=np.int32)])
        self._times = np.concatenate([self._times, times])
        self._next_id += len(ids)
        self._next_key += 1

        record = VideoRecord(
            video_id=video_id,
            key=key,
            source_path=source_path,
            sha256=sha256,
            duration_s=round(float(duration_s), 3),
            sample_fps=float(sample_fps),
            n_vectors=len(ids),
            indexed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._videos[key] = record
        return record

    def remove_video(self, video_id: str) -> VideoRecord:
        record = self.get(video_id)
        if record is None:
            raise LibraryError(f"video id {video_id!r} is not in the library")
        mask = self._keys == record.key
        removed = self._index.remove_ids(faiss.IDSelectorBatch(self._ids[mask]))
        if removed != record.n_vectors:
            raise LibraryError(f"index removed {removed} vectors, expected {record.n_vectors}")
        keep = ~mask
        self._ids, self._keys, self._times = self._ids[keep], self._keys[keep], self._times[keep]
        del self._videos[record.key]
        return record

    # ------------------------------------------------------------- persistence

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        return (Path(directory) / _MANIFEST).is_file()

    @classmethod
    def load(cls, directory: str | Path) -> "Library":
        directory = Path(directory)
        try:
            manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LibraryError(f"no library at {directory}") from None
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise LibraryError(
                f"library schema {manifest.get('schema_version')} is not supported "
                f"(expected {SCHEMA_VERSION}); rebuild it"
            )
        lib = cls(manifest["embedder"], int(manifest["dim"]))
        lib._next_id = int(manifest["next_id"])
        lib._next_key = int(manifest["next_key"])
        raw = np.fromfile(directory / _INDEX, dtype=np.uint8)
        lib._index = faiss.deserialize_index(raw)
        with np.load(directory / _ROWS) as rows:
            lib._ids, lib._keys, lib._times = rows["ids"], rows["keys"], rows["times"]
        videos = json.loads((directory / _VIDEOS).read_text(encoding="utf-8"))
        lib._videos = {v["key"]: VideoRecord(**v) for v in videos}
        lib._check_consistent()
        return lib

    @classmethod
    def open_or_create(cls, directory: str | Path, embedder: str, dim: int) -> "Library":
        if not cls.exists(directory):
            return cls(embedder, dim)
        lib = cls.load(directory)
        if lib.embedder != embedder or lib.dim != dim:
            raise LibraryError(
                f"library at {directory} was built with {lib.embedder} (dim {lib.dim}), "
                f"not {embedder} (dim {dim}); use a separate --library directory per embedder"
            )
        return lib

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._check_consistent()
        index_bytes = faiss.serialize_index(self._index)
        _atomic_write(directory / _INDEX, lambda f: f.write(index_bytes.tobytes()))
        _atomic_write(
            directory / _ROWS,
            lambda f: np.savez(f, ids=self._ids, keys=self._keys, times=self._times),
        )
        videos = [asdict(v) for v in self.videos()]
        _atomic_write(directory / _VIDEOS, lambda f: f.write(_json_bytes(videos)))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "embedder": self.embedder,
            "dim": self.dim,
            "metric": "inner_product",
            "next_id": self._next_id,
            "next_key": self._next_key,
            "n_videos": self.n_videos,
            "n_vectors": self.n_vectors,
        }
        _atomic_write(directory / _MANIFEST, lambda f: f.write(_json_bytes(manifest)))

    # ---------------------------------------------------------------- internal

    def _check_vectors(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise LibraryError(f"expected vectors of shape (n, {self.dim}), got {x.shape}")
        if not np.isfinite(x).all():
            raise LibraryError("vectors contain NaN or inf")
        return x

    def _check_consistent(self) -> None:
        n = self.n_vectors
        if not (len(self._ids) == len(self._keys) == len(self._times) == n):
            raise LibraryError(
                f"library is inconsistent: index has {n} vectors, rows have {len(self._ids)}"
            )
        if sum(v.n_vectors for v in self._videos.values()) != n:
            raise LibraryError("library is inconsistent: video records don't add up to the index")
        if n and not set(np.unique(self._keys).tolist()) <= set(self._videos):
            raise LibraryError("library is inconsistent: vectors reference unknown videos")


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2).encode("utf-8")


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
