from ..config import Settings
from .types import Algorithm, Candidate, Decision, Evidence, MatchResult
from .v1 import V1Voting

ALGORITHMS = ["v1"]


def create_algorithm(settings: Settings) -> Algorithm:
    if settings.algorithm == "v1":
        return V1Voting(min_conf=settings.v1_min_conf, min_vote_ratio=settings.v1_min_vote_ratio)
    raise ValueError(f"unknown algorithm {settings.algorithm!r}")


__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "Candidate",
    "Decision",
    "Evidence",
    "MatchResult",
    "V1Voting",
    "create_algorithm",
]
