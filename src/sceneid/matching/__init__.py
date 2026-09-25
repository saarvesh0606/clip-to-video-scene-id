from ..config import Settings
from .types import Algorithm, Candidate, Decision, Evidence, MatchResult
from .v1 import V1Voting
from .v2 import OffsetVoting

ALGORITHMS = ["v1", "v2"]


def create_algorithm(settings: Settings) -> Algorithm:
    if settings.algorithm == "v1":
        return V1Voting(min_conf=settings.v1_min_conf, min_vote_ratio=settings.v1_min_vote_ratio)
    if settings.algorithm == "v2":
        return OffsetVoting(
            min_score=settings.v2_min_score,
            max_ratio=settings.v2_max_ratio,
            tolerance_s=settings.v2_tolerance_s,
        )
    raise ValueError(f"unknown algorithm {settings.algorithm!r}")


__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "Candidate",
    "Decision",
    "Evidence",
    "MatchResult",
    "OffsetVoting",
    "V1Voting",
    "create_algorithm",
]
