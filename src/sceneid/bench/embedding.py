"""The expensive stage: embedding the library films and every query clip.

This is the only step that needs a GPU to be fast. Query clips are rendered, sampled and
deleted on the fly by worker threads while the main thread embeds frames in large batches,
so disk use stays small. Results are written in shards as they complete, and a restarted
run skips every query already stored, so a dropped Colab session loses at most one shard.
"""

import json
import logging
import os
import platform
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import Settings
from ..embedders import Embedder
from ..frames import VideoError, probe, sample_frames
from ..indexer import index_video
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


def build_library(
    manifest: Manifest,
    films: dict[str, FilmFile],
    embedder: Embedder,
    settings: Settings,
    out_dir: str | Path,
) -> Library:
    """Index every library film, saving after each one; films already indexed are skipped."""
    library = Library.open_or_create(out_dir, embedder.name, embedder.dim)
    for film in manifest.library:
        if library.get(film.id):
            log.info("bench.index_skip", extra={"film": film.id})
            continue
        index_video(
            library,
            embedder,
            films[film.id].path,
            video_id=film.id,
            fps=settings.index_fps,
            short_side=settings.frame_short_side,
            batch_size=settings.batch_size,
        )
        library.save(out_dir)
    return library


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


# ------------------------------------------------------------------------ query embedding


def _prepare(spec: QuerySpec, film: FilmFile, settings: Settings, tmp_dir: Path):
    """Worker thread: render the clip, sample its frames, delete it."""
    clip = tmp_dir / f"{spec.query_id}.mp4"
    started = time.perf_counter()
    render(
        spec.distortion,
        film.path,
        spec.ref_start_s,
        spec.ref_duration_s,
        clip,
        seed=spec.seed,
        width=film.width,
        height=film.height,
    )
    render_ms = (time.perf_counter() - started) * 1e3
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
    embedder: Embedder,
    settings: Settings,
    out_dir: str | Path,
    *,
    queries_digest: str,
    workers: int = 2,
    shard_size: int = 200,
    limit: int | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _store_meta(embedder, settings, queries_digest)
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        different = {k: (existing.get(k), v) for k, v in meta.items() if existing.get(k) != v}
        if different:
            raise ValueError(
                f"{out_dir} holds embeddings made with other settings {different}; "
                "use a new directory"
            )
    else:
        meta_path.write_text(
            json.dumps({**meta, "machine": machine_info(), "sceneid": __version__}, indent=2)
        )

    store = load_query_store(out_dir)
    todo = [s for s in specs if s.query_id not in store.queries and s.query_id not in store.failed]
    if limit is not None:
        todo = todo[:limit]
    log.info(
        "bench.embed_start",
        extra={
            "total": len(specs),
            "done": len(store.queries),
            "todo": len(todo),
            "workers": workers,
        },
    )
    failed = dict(store.failed)
    pending_shard: list[dict] = []
    batch: list[dict] = []
    started = time.perf_counter()
    n_done = 0

    def flush_batch() -> None:
        if not batch:
            return
        frames = [f for row in batch for f in row["frames"]]
        t0 = time.perf_counter()
        vectors = embedder.embed(frames)
        per_frame_ms = (time.perf_counter() - t0) * 1e3 / len(frames)
        offset = 0
        for row in batch:
            n = len(row.pop("frames"))
            row["vectors"] = vectors[offset : offset + n]
            row["embed_ms"] = per_frame_ms * n
            offset += n
            pending_shard.append(row)
        batch.clear()

    def flush_shard() -> None:
        if pending_shard:
            _write_shard(out_dir, pending_shard)
            pending_shard.clear()
        (out_dir / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    with (
        tempfile.TemporaryDirectory(prefix="sceneid-bench-") as tmp,
        ThreadPoolExecutor(workers) as pool,
    ):
        tmp_dir = Path(tmp)
        queue: deque = deque()
        specs_iter = iter(todo)

        def submit_next() -> None:
            spec = next(specs_iter, None)
            if spec is not None:
                queue.append(
                    (spec, pool.submit(_prepare, spec, films[spec.film_id], settings, tmp_dir))
                )

        for _ in range(workers * 3):
            submit_next()
        while queue:
            spec, future = queue.popleft()
            submit_next()
            try:
                batch.append(future.result())
            except (RenderError, VideoError, OSError, ValueError) as exc:
                failed[spec.query_id] = str(exc)
                log.warning(
                    "bench.query_failed", extra={"query_id": spec.query_id, "error": str(exc)}
                )
            n_done += 1
            if sum(len(r["frames"]) for r in batch) >= settings.batch_size:
                flush_batch()
            if len(pending_shard) >= shard_size:
                flush_shard()
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
        flush_shard()
    return {"embedded": n_done - (len(failed) - len(store.failed)), "failed": len(failed)}
