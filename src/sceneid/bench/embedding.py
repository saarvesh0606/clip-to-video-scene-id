"""The expensive stage: embedding the library films and every query clip.

Each film and each query clip is decoded once and its frames are fed to every embedder in
the run, so comparing models costs one decoding pass, not one per model.

Query clips are rendered by worker threads while the main thread embeds frames in large
batches. Rendering (ffmpeg, on the CPU) is by far the slowest step, so rendered clips can be
kept in a clips directory: later runs, for example with a new embedder, reuse them instead
of rendering again. Results are written in shards per embedder as they complete, and a
restarted run skips every query already stored, so a dropped Colab session loses at most one
shard.
"""

import json
import logging
import os
import platform
import tempfile
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import Settings
from ..embedders import Embedder
from ..frames import VideoError, probe, sample_frames
from ..indexer import embed_video_multi
from ..library import Library
from ..matcher import QueryEmbedding, query_sample_fps
from .distortions import RenderError, render
from .download import FilmFile
from .manifest import Manifest
from .queries import QuerySpec

log = logging.getLogger(__name__)

_SHARD_GLOB = "shard-*.npz"


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


# ------------------------------------------------------------------------------- library


def build_libraries(
    manifest: Manifest,
    films: dict[str, FilmFile],
    embedders: Sequence[Embedder],
    settings: Settings,
    out_dirs: dict[str, Path],
) -> dict[str, Library]:
    """Index every library film into one library per embedder, decoding each film once.

    Each library is saved after every film, and films a library already holds are skipped,
    so an interrupted run resumes where it stopped.
    """
    libraries = {e.name: Library.open_or_create(out_dirs[e.name], e.name, e.dim) for e in embedders}
    for film in manifest.library:
        needed = [e for e in embedders if libraries[e.name].get(film.id) is None]
        if not needed:
            log.info("bench.index_skip", extra={"film": film.id})
            continue
        info = films[film.id]
        out = embed_video_multi(
            needed,
            Path(info.path),
            fps=settings.index_fps,
            short_side=settings.frame_short_side,
            batch_size=settings.batch_size,
        )
        for e in needed:
            libraries[e.name].add_video(
                film.id,
                out.vectors[e.name],
                out.times,
                source_path=info.path,
                sha256=info.sha256,
                duration_s=info.duration_s,
                sample_fps=settings.index_fps,
            )
            libraries[e.name].save(out_dirs[e.name])
        log.info(
            "bench.index_film_done",
            extra={
                "film": film.id,
                "frames": len(out.times),
                "decode_s": round(out.decode_s, 1),
                "embed_s": {k: round(v, 1) for k, v in out.embed_s.items()},
            },
        )
    return libraries


def build_library(manifest, films, embedder, settings, out_dir) -> Library:
    """One embedder's library; see build_libraries."""
    return build_libraries(manifest, films, [embedder], settings, {embedder.name: Path(out_dir)})[
        embedder.name
    ]


# --------------------------------------------------------------------------- query store


def _store_meta(embedder: Embedder, settings: Settings, queries_digest: str) -> dict:
    return {
        "embedder": embedder.name,
        "dim": embedder.dim,
        "query_fps": settings.query_fps,
        "query_max_frames": settings.query_max_frames,
        "frame_short_side": settings.frame_short_side,
        "queries_digest": queries_digest,
    }


@dataclass
class QueryStore:
    meta: dict
    queries: dict[str, QueryEmbedding] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)  # query_id -> error


def load_query_store(directory: str | Path) -> QueryStore:
    directory = Path(directory)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no query embeddings at {directory}")
    store = QueryStore(json.loads(meta_path.read_text(encoding="utf-8")))
    for shard in sorted(directory.glob(_SHARD_GLOB)):
        with np.load(shard) as z:
            ids, starts, counts = z["ids"], z["starts"], z["counts"]
            times, vectors = z["times"], z["vectors"]
            decode_ms, embed_ms, render_ms = z["decode_ms"], z["embed_ms"], z["render_ms"]
        for i, qid in enumerate(ids.tolist()):
            sl = slice(int(starts[i]), int(starts[i]) + int(counts[i]))
            store.queries[qid] = QueryEmbedding(
                times[sl],
                vectors[sl],
                {
                    "render": float(render_ms[i]),
                    "decode": float(decode_ms[i]),
                    "embed": float(embed_ms[i]),
                },
            )
    failed_path = directory / "failed.json"
    if failed_path.is_file():
        store.failed = json.loads(failed_path.read_text(encoding="utf-8"))
    return store


def _write_shard(directory: Path, rows: list[dict]) -> None:
    index = len(list(directory.glob(_SHARD_GLOB)))
    counts = np.array([len(r["times"]) for r in rows], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    tmp = directory / f"shard-{index:05d}.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            ids=np.array([r["query_id"] for r in rows]),
            starts=starts,
            counts=counts,
            times=np.concatenate([r["times"] for r in rows]).astype(np.float32),
            vectors=np.concatenate([r["vectors"] for r in rows]).astype(np.float32),
            render_ms=np.array([r["render_ms"] for r in rows], np.float32),
            decode_ms=np.array([r["decode_ms"] for r in rows], np.float32),
            embed_ms=np.array([r["embed_ms"] for r in rows], np.float32),
        )
    os.replace(tmp, directory / f"shard-{index:05d}.npz")


class _StoreWriter:
    """One embedder's query store during a run: what it still needs, and what to write."""

    def __init__(self, embedder: Embedder, directory: Path, meta: dict):
        self.embedder = embedder
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "meta.json"
        if meta_path.is_file():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            different = {k: (existing.get(k), v) for k, v in meta.items() if existing.get(k) != v}
            if different:
                raise ValueError(
                    f"{directory} holds embeddings made with other settings {different}; "
                    "use a new directory"
                )
        else:
            meta_path.write_text(
                json.dumps({**meta, "machine": machine_info(), "sceneid": __version__}, indent=2)
            )
        store = load_query_store(directory)
        self.done = set(store.queries)
        self.failed = dict(store.failed)
        self.pending: list[dict] = []

    def needs(self, query_id: str) -> bool:
        return query_id not in self.done and query_id not in self.failed

    def flush(self) -> None:
        if self.pending:
            _write_shard(self.dir, self.pending)
            self.done.update(r["query_id"] for r in self.pending)
            self.pending.clear()
        (self.dir / "failed.json").write_text(json.dumps(self.failed, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------ query embedding


def _obtain_clip(spec: QuerySpec, film: FilmFile, clips_dir: Path) -> tuple[Path, float]:
    """The rendered clip, from the cache if it's there. Returns (path, render_ms or NaN)."""
    clip = clips_dir / f"{spec.query_id}.mp4"
    if clip.is_file() and clip.stat().st_size > 0:
        return clip, float("nan")
    started = time.perf_counter()
    part = clips_dir / f"{spec.query_id}.part.mp4"
    render(
        spec.distortion,
        film.path,
        spec.ref_start_s,
        spec.ref_duration_s,
        part,
        seed=spec.seed,
        width=film.width,
        height=film.height,
    )
    os.replace(part, clip)  # a clip in the cache is always complete
    return clip, (time.perf_counter() - started) * 1e3


def _prepare(spec: QuerySpec, film: FilmFile, settings: Settings, clips_dir: Path, keep: bool):
    """Worker thread: get the rendered clip, sample its frames, drop it unless kept."""
    clip, render_ms = _obtain_clip(spec, film, clips_dir)
    try:
        started = time.perf_counter()
        fps = query_sample_fps(
            probe(clip).duration_s, settings.query_fps, settings.query_max_frames
        )
        sampled = list(
            sample_frames(
                clip,
                fps,
                short_side=settings.frame_short_side,
                max_frames=settings.query_max_frames,
            )
        )
        decode_ms = (time.perf_counter() - started) * 1e3
    finally:
        if not keep:
            clip.unlink(missing_ok=True)
    if not sampled:
        raise RenderError("no frames decoded from the rendered clip")
    return {
        "query_id": spec.query_id,
        "times": np.array([t for t, _ in sampled], np.float32),
        "frames": [f for _, f in sampled],
        "render_ms": render_ms,
        "decode_ms": decode_ms,
    }


def embed_queries(
    specs: list[QuerySpec],
    films: dict[str, FilmFile],
    embedders: Sequence[Embedder] | Embedder,
    settings: Settings,
    out_dirs: dict[str, Path] | str | Path,
    *,
    queries_digest: str,
    clips_dir: str | Path | None = None,
    workers: int = 2,
    shard_size: int = 200,
    limit: int | None = None,
) -> dict:
    """Embed every query with every embedder. Pass `clips_dir` to keep rendered clips there.

    For one embedder, `embedders` may be a single Embedder and `out_dirs` a single path.
    """
    if not isinstance(embedders, Sequence):
        embedders = [embedders]
        out_dirs = {embedders[0].name: Path(out_dirs)}
    writers = [
        _StoreWriter(e, Path(out_dirs[e.name]), _store_meta(e, settings, queries_digest))
        for e in embedders
    ]
    todo = [s for s in specs if any(w.needs(s.query_id) for w in writers)]
    if limit is not None:
        todo = todo[:limit]
    log.info(
        "bench.embed_start",
        extra={
            "total": len(specs),
            "todo": len(todo),
            "embedders": [e.name for e in embedders],
            "workers": workers,
            "clip_cache": str(clips_dir) if clips_dir else None,
        },
    )
    batch: list[dict] = []
    started = time.perf_counter()
    n_done = 0

    def flush_batch() -> None:
        if not batch:
            return
        for w in writers:
            rows = [r for r in batch if w.needs(r["query_id"])]
            if not rows:
                continue
            frames = [f for r in rows for f in r["frames"]]
            t0 = time.perf_counter()
            vectors = w.embedder.embed(frames)
            per_frame_ms = (time.perf_counter() - t0) * 1e3 / len(frames)
            offset = 0
            for r in rows:
                n = len(r["frames"])
                w.pending.append(
                    {
                        "query_id": r["query_id"],
                        "times": r["times"],
                        "vectors": vectors[offset : offset + n],
                        "render_ms": r["render_ms"],
                        "decode_ms": r["decode_ms"],
                        "embed_ms": per_frame_ms * n,
                    }
                )
                offset += n
        batch.clear()

    with tempfile.TemporaryDirectory(prefix="sceneid-bench-") as tmp:
        keep = clips_dir is not None
        cache = Path(clips_dir) if keep else Path(tmp)
        cache.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as pool:
            queue: deque = deque()
            specs_iter = iter(todo)

            def submit_next() -> None:
                spec = next(specs_iter, None)
                if spec is not None:
                    film = films[spec.film_id]
                    queue.append((spec, pool.submit(_prepare, spec, film, settings, cache, keep)))

            for _ in range(workers * 3):
                submit_next()
            while queue:
                spec, future = queue.popleft()
                submit_next()
                try:
                    batch.append(future.result())
                except (RenderError, VideoError, OSError, ValueError) as exc:
                    for w in writers:
                        if w.needs(spec.query_id):
                            w.failed[spec.query_id] = str(exc)
                    log.warning(
                        "bench.query_failed", extra={"query_id": spec.query_id, "error": str(exc)}
                    )
                n_done += 1
                if sum(len(r["frames"]) for r in batch) >= settings.batch_size:
                    flush_batch()
                for w in writers:
                    if len(w.pending) >= shard_size:
                        w.flush()
                if n_done % 50 == 0 or n_done == len(todo):
                    rate = n_done / (time.perf_counter() - started)
                    log.info(
                        "bench.embed_progress",
                        extra={
                            "done": n_done,
                            "todo": len(todo),
                            "per_s": round(rate, 2),
                            "eta_min": round((len(todo) - n_done) / max(rate, 1e-9) / 60, 1),
                        },
                    )
            flush_batch()
            for w in writers:
                w.flush()
    return {
        "processed": n_done,
        "failed": {w.embedder.name: len(w.failed) for w in writers},
    }
