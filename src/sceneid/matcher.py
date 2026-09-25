"""Identifying a query clip: sample, embed, search, decide.

Embedding is split from deciding so benchmarks can embed each clip once and then replay
many algorithms and thresholds over the cached vectors.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings
from .embedders import Embedder, create_embedder
from .frames import VideoError, probe
from .indexer import embed_video
from .library import Library
from .matching import Algorithm, Candidate, Evidence, MatchResult, create_algorithm

log = logging.getLogger(__name__)


@dataclass
class QueryEmbedding:
    times: np.ndarray  # (n,) seconds into the clip
    vectors: np.ndarray  # (n, dim)
    timings_ms: dict[str, float] = field(default_factory=dict)


class Matcher:
    def __init__(
        self,
        embedder: Embedder,
        library: Library,
        algorithm: Algorithm,
        *,
        query_fps: float,
        max_frames: int,
        top_k: int,
        short_side: int | None,
        batch_size: int = 32,
    ):
        if library.embedder != embedder.name:
            raise ValueError(
                f"library was built with {library.embedder}, but the embedder is {embedder.name}"
            )
        self.embedder = embedder
        self.library = library
        self.algorithm = algorithm
        self.query_fps = query_fps
        self.max_frames = max_frames
        self.top_k = top_k
        self.short_side = short_side
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder | None = None) -> "Matcher":
        embedder = embedder or create_embedder(
            settings.embedder, device=settings.device, batch_size=settings.batch_size
        )
        library = Library.load(settings.library_dir)
        return cls(
            embedder,
            library,
            create_algorithm(settings),
            query_fps=settings.query_fps,
            max_frames=settings.query_max_frames,
            top_k=settings.top_k,
            short_side=settings.frame_short_side,
            batch_size=settings.batch_size,
        )

    def embed_clip(self, path: str | Path) -> QueryEmbedding:
        # Spread at most `max_frames` over the whole clip rather than using its first
        # `max_frames / query_fps` seconds. Clips short enough to fit are sampled as before.
        duration = probe(path).duration_s
        fps = min(self.query_fps, self.max_frames / duration) if duration > 0 else self.query_fps
        times, vectors, decode_s, embed_s = embed_video(
            self.embedder,
            Path(path),
            fps=fps,
            short_side=self.short_side,
            batch_size=self.batch_size,
            max_frames=self.max_frames,
        )
        if len(times) == 0:
            raise VideoError("no frames could be decoded from the clip")
        return QueryEmbedding(
            times, vectors, {"decode": round(decode_s * 1e3, 1), "embed": round(embed_s * 1e3, 1)}
        )

    def decide(self, query: QueryEmbedding) -> MatchResult:
        started = time.perf_counter()
        hits = self.library.search(query.vectors, self.top_k)
        search_ms = (time.perf_counter() - started) * 1e3

        started = time.perf_counter()
        d = self.algorithm.decide(query.times, hits)
        decide_ms = (time.perf_counter() - started) * 1e3

        timings = {**query.timings_ms, "search": round(search_ms, 2), "decide": round(decide_ms, 2)}
        timings["total"] = round(sum(timings.values()), 1)
        candidate = None
        if d.candidate_key is not None:
            offset = round(d.offset_s, 3) if d.offset_s is not None else None
            candidate = Candidate(video_id=self.library.video_id(d.candidate_key), offset_s=offset)
        return MatchResult(
            status="match" if d.accepted else "unknown",
            video_id=candidate.video_id if d.accepted else None,
            confidence=d.confidence,
            offset_s=candidate.offset_s if d.accepted else None,
            ref_window=d.window if d.accepted else None,
            candidate=candidate,
            reason=d.reason,
            votes={self.library.video_id(k): n for k, n in d.votes.items()},
            evidence=[
                Evidence(query_t=round(q, 3), ref_t=round(r, 3), score=round(s, 4))
                for q, r, s in d.evidence
            ],
            diagnostics=d.diagnostics,
            algorithm=self.algorithm.name,
            embedder=self.embedder.name,
            n_query_frames=len(query.times),
            timings_ms=timings,
        )

    def identify(self, path: str | Path) -> MatchResult:
        result = self.decide(self.embed_clip(path))
        log.info(
            "match.done",
            extra={
                "status": result.status,
                "video_id": result.video_id,
                "candidate": result.candidate.video_id if result.candidate else None,
                "confidence": result.confidence,
                "offset_s": result.offset_s,
                "frames": result.n_query_frames,
                "timings_ms": result.timings_ms,
            },
        )
        return result
