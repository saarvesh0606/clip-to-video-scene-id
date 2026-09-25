"""V2: pick the video whose frames line up with the clip in time.

A clip cut from a video matches it at one consistent offset (reference time minus query
time). V2 searches for that offset directly:

1. Every neighbour of every query frame proposes an offset for its video.
2. For each video and candidate offset, each query frame contributes the similarity of its
   best neighbour within ``tolerance_s`` of that offset. It contributes once at most, so
   ten neighbours from one static shot can't stack up votes.
3. A video's score is the mean over *all* query frames, with frames that don't line up
   adding zero. It reads as "the clip's average similarity at its best alignment", in [0, 1].
4. The best video must beat the runner-up by a margin (the ratio test): a clip that fits
   two videos equally well is ambiguous, however high its score.
5. The offset is the similarity-weighted median of the aligned frames' offsets, so it
   isn't rounded to a histogram bin.

Compared with V1 (see ``v1.py``): alignment chooses the video instead of following raw
vote counts, frames that don't line up count for nothing (there is no fallback), and the
ratio test doesn't get stricter as the library grows.

The default thresholds are provisional. They will be replaced by values chosen from the
benchmark's validation ROC curve.
"""

from dataclasses import dataclass

import numpy as np

from ..library import SearchHits
from .types import Decision


@dataclass
class OffsetVoting:
    min_score: float = 0.6
    max_ratio: float = 0.9  # runner-up score / best score
    tolerance_s: float = 0.6
    max_evidence: int = 10
    name: str = "v2"

    def decide(self, query_times: np.ndarray, hits: SearchHits) -> Decision:
        keys = hits.video_keys
        valid = keys >= 0
        if not valid.any():
            return Decision(False, None, 0.0, None, None, "no_candidates")

        query_times = np.asarray(query_times, dtype=np.float64)
        offsets = hits.ref_times.astype(np.float64) - query_times[:, None]
        scores = np.clip(hits.scores.astype(np.float64), 0.0, 1.0)

        fits = [self._fit_video(int(k), offsets, scores, keys == k) for k in np.unique(keys[valid])]
        fits.sort(key=lambda f: f.score, reverse=True)
        best = fits[0]
        runner_up = fits[1].score if len(fits) > 1 else 0.0
        ratio = runner_up / best.score if best.score > 0 else 1.0

        aligned = best.frame_scores > 0
        offset = _weighted_median(best.frame_offsets[aligned], best.frame_scores[aligned])
        spread = float(np.median(np.abs(best.frame_offsets[aligned] - offset)))
        q_aligned = query_times[aligned]
        window = (
            round(offset + float(q_aligned.min()), 2),
            round(offset + float(q_aligned.max()), 2),
        )

        order = np.argsort(-best.frame_scores[aligned])[: self.max_evidence]
        idx = np.flatnonzero(aligned)[order]
        evidence = sorted(
            (
                float(query_times[i]),
                float(query_times[i] + best.frame_offsets[i]),
                float(best.frame_scores[i]),
            )
            for i in idx
        )

        if best.score < self.min_score:
            accepted, reason = False, "low_score"
        elif ratio > self.max_ratio:
            accepted, reason = False, "ambiguous"
        else:
            accepted, reason = True, None

        return Decision(
            accepted=accepted,
            candidate_key=best.key,
            confidence=round(best.score, 4),
            offset_s=offset,
            window=window,
            reason=reason,
            votes={f.key: f.n_aligned for f in fits},
            evidence=evidence,
            diagnostics={
                "runner_up_score": round(runner_up, 4),
                "ratio": round(ratio, 4),
                "coverage": round(float(aligned.mean()), 4),
                "offset_spread_s": round(spread, 3),
            },
        )

    def _fit_video(self, key: int, offsets: np.ndarray, scores: np.ndarray, mask: np.ndarray):
        """The best offset for one video, and each query frame's contribution at it."""
        n = offsets.shape[0]
        candidates = np.unique(np.round(offsets[mask], 2))
        near = np.abs(offsets[None] - candidates[:, None, None]) <= self.tolerance_s
        contrib = np.where(near & mask[None], scores[None], 0.0)  # (candidates, n, k)
        per_frame = contrib.max(axis=2)
        support = per_frame.sum(axis=1) / n
        c = int(support.argmax())
        best_rank = contrib[c].argmax(axis=1)
        return _Fit(
            key=key,
            score=float(support[c]),
            frame_scores=per_frame[c],
            frame_offsets=offsets[np.arange(n), best_rank],
        )


@dataclass
class _Fit:
    key: int
    score: float
    frame_scores: np.ndarray  # (n_query,) 0 where the frame doesn't line up
    frame_offsets: np.ndarray  # (n_query,)

    @property
    def n_aligned(self) -> int:
        return int((self.frame_scores > 0).sum())


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cumulative[-1] / 2)])
