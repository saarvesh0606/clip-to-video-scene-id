"""The V1 decision rule, ported unchanged so benchmarks have an honest baseline.

This reproduces ``aggregate_votes`` and ``apply_unknown_gate`` from the original
``query/match_local.py`` (commit f9fb28f) exactly; ``tests/test_v1_parity.py`` checks the
port against that code on random inputs. Its known flaws are kept on purpose, because
the point of this module is to measure them:

* The video is chosen by raw vote count over all top-k ranks, before any temporal check.
* When too few hits agree on an offset, it falls back to *all* hits as inliers, which
  sets ``align_ratio`` to 1.0: a clip with no temporal consistency gets the maximum
  alignment score.
* ``min_vote_ratio`` counts every top-k neighbour, so it gets harder to pass as the
  library grows, whether or not the answer is right.

Do not fix these here. Improvements belong in a new algorithm, measured against this one.
"""

from dataclasses import dataclass

import numpy as np

from ..library import SearchHits
from .types import Decision


@dataclass
class V1Voting:
    min_conf: float = 0.83
    min_vote_ratio: float = 0.90
    window_size: float = 6.0
    bin_size: float = 0.5
    inlier_tol: float = 0.75
    strong_score: float = 0.90
    name: str = "v1"

    gate_params = ("min_conf", "min_vote_ratio")

    @staticmethod
    def gate(confidence, diagnostics, min_conf, min_vote_ratio):
        """Accept or reject. Works elementwise on arrays, so benchmarks can sweep thresholds."""
        return (confidence >= min_conf) & (diagnostics["vote_ratio"] >= min_vote_ratio)

    def decide(self, query_times: np.ndarray, hits: SearchHits) -> Decision:
        n_query, top_k = hits.scores.shape

        votes: dict[int, int] = {}
        score_sum: dict[int, float] = {}
        hits_all = []
        for qi in range(n_query):
            t_q = float(query_times[qi])
            for r in range(top_k):
                key = int(hits.video_keys[qi, r])
                if key < 0:
                    continue
                score = float(hits.scores[qi, r])
                ts = float(hits.ref_times[qi, r])
                votes[key] = votes.get(key, 0) + 1
                score_sum[key] = score_sum.get(key, 0.0) + score
                hits_all.append({"key": key, "ts": ts, "t_q": t_q, "score": score, "rank": r})

        if not votes:
            return Decision(False, None, 0.0, None, None, "no_candidates")

        best = sorted(votes, key=lambda v: (votes[v], score_sum[v]), reverse=True)[0]
        best_hits = [h for h in hits_all if h["key"] == best]
        best_hits.sort(key=lambda h: h["score"], reverse=True)

        top1 = [h["score"] for h in best_hits if h["rank"] == 0]
        if not top1:
            base_conf = 0.0
        else:
            top1_arr = np.array(top1, dtype=np.float32)
            base_conf = 0.7 * float(top1_arr.mean()) + 0.3 * float(
                (top1_arr >= self.strong_score).mean()
            )

        # Temporal alignment: histogram of (reference time - query time).
        offsets = np.array([h["ts"] - h["t_q"] for h in best_hits], dtype=np.float32)
        bins = np.floor(offsets / self.bin_size).astype(int)
        bin_counts: dict = {}
        for b in bins:
            bin_counts[b] = bin_counts.get(b, 0) + 1
        best_bin = max(bin_counts, key=bin_counts.get)
        best_offset = float((best_bin + 0.5) * self.bin_size)

        inliers = [
            h for h in best_hits if abs((h["ts"] - h["t_q"]) - best_offset) <= self.inlier_tol
        ]
        aligned = len(inliers) >= max(5, int(0.3 * len(best_hits)))
        if not aligned:
            inliers = best_hits  # the V1 fallback: see the module docstring
        align_ratio = len(inliers) / max(1, len(best_hits))

        confidence = round(float(np.clip(0.6 * base_conf + 0.4 * align_ratio, 0.0, 1.0)), 4)

        times_est = np.array([h["t_q"] + best_offset for h in inliers], dtype=np.float32)
        scores_est = np.array([h["score"] for h in inliers], dtype=np.float32)
        w = np.maximum(scores_est.astype(np.float64), 1e-6)
        est_timestamp = round(float((times_est.astype(np.float64) * w).sum() / float(w.sum())), 2)
        window = self._densest_window(times_est, scores_est)

        seen = set()
        evidence = []
        for h in best_hits:
            ref = round(h["ts"], 2)
            if ref in seen:
                continue
            seen.add(ref)
            evidence.append((h["t_q"], h["ts"], h["score"]))
            if len(evidence) == 10:
                break

        total_votes = sum(votes.values())
        vote_ratio = votes[best] / total_votes if total_votes else 0.0
        diagnostics = {
            "base_conf": base_conf,
            "align_ratio": align_ratio,
            "aligned": int(aligned),
            "vote_ratio": vote_ratio,
            "est_timestamp": est_timestamp,
        }
        accepted = bool(self.gate(confidence, diagnostics, self.min_conf, self.min_vote_ratio))
        reason = None if accepted else "below_threshold"

        return Decision(
            accepted=accepted,
            candidate_key=best,
            confidence=confidence,
            offset_s=best_offset,
            window=window,
            reason=reason,
            votes=votes,
            evidence=evidence,
            diagnostics=diagnostics,
        )

    def _densest_window(self, times: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        """The `window_size`-second span holding the most inliers (ties: higher score sum)."""
        order = np.argsort(times)
        times_sorted, scores_sorted = times[order], scores[order]
        best_i, best_j = 0, 1
        best_count = 1
        best_score_sum = float(scores_sorted[0])
        j = 0
        for i in range(len(times_sorted)):
            while j < len(times_sorted) and times_sorted[j] <= times_sorted[i] + self.window_size:
                j += 1
            count = j - i
            window_score = float(scores_sorted[i:j].sum()) if j > i else 0.0
            if count > best_count or (count == best_count and window_score > best_score_sum):
                best_i, best_j, best_count, best_score_sum = i, j, count, window_score
        cluster = times_sorted[best_i:best_j]
        if len(cluster) == 0:
            cluster = times_sorted
        return round(float(cluster[0]), 2), round(float(cluster[-1]), 2)
