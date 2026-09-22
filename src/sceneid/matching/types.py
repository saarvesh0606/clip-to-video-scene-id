from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

from ..library import SearchHits


@dataclass
class Decision:
    """What an algorithm concluded from the search hits, in library keys."""

    accepted: bool
    candidate_key: int | None  # the best-scoring video, reported even when rejected
    confidence: float
    offset_s: float | None  # where the clip starts in the candidate video
    window: tuple[float, float] | None
    reason: str | None
    votes: dict[int, int] = field(default_factory=dict)
    evidence: list[tuple[float, float, float]] = field(default_factory=list)  # (q_t, ref_t, s)
    diagnostics: dict[str, float | int | str | None] = field(default_factory=dict)


class Algorithm(Protocol):
    name: str

    def decide(self, query_times: np.ndarray, hits: SearchHits) -> Decision: ...


class Evidence(BaseModel):
    query_t: float = Field(description="Seconds into the query clip")
    ref_t: float = Field(description="Seconds into the matched video")
    score: float = Field(description="Cosine similarity of the two frames")


class Candidate(BaseModel):
    video_id: str
    offset_s: float | None


class MatchResult(BaseModel):
    status: Literal["match", "unknown"]
    video_id: str | None = Field(description="The matched video, or null when unknown")
    confidence: float
    offset_s: float | None = Field(description="Where the clip starts in the matched video")
    ref_window: tuple[float, float] | None = Field(
        description="The span of the matched video the evidence covers"
    )
    candidate: Candidate | None = Field(
        description="The best guess and its offset, reported even when the answer is unknown"
    )
    reason: str | None
    votes: dict[str, int]
    evidence: list[Evidence]
    diagnostics: dict[str, float | int | str | None]
    algorithm: str
    embedder: str
    n_query_frames: int
    timings_ms: dict[str, float]
