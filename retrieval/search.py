# retrieval/search.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import faiss


def empty_result(reason: str) -> Dict[str, Any]:
    return {
        "best_video_id": None,
        "confidence": 0.0,
        "est_timestamp": None,
        "time_window": [None, None],
        "votes_per_video": {},
        "score_sum_per_video": {},
        "top_evidence": [],
        "reason": reason,
    }


def faiss_scores_to_similarity(D: np.ndarray, index) -> np.ndarray:
    """
    EXACT V1 behavior:
    - If L2 metric: sim = 1 - D/4 (clipped [0,1])
    - Else: return D unchanged
    """
    try:
        metric = index.metric_type
    except Exception:
        return D

    if metric == faiss.METRIC_L2:
        sim = 1.0 - (D / 4.0)
        return np.clip(sim, 0.0, 1.0)
    return D


def aggregate_votes(
    I: np.ndarray,
    D_sim: np.ndarray,
    meta: dict,
    query_times: np.ndarray | None = None,
    window_size: float = 6.0,
    bin_size: float = 0.5,
    inlier_tol: float = 0.75,
):
    """
    EXACT V1 behavior (copied).
    """
    n_query, top_k = I.shape

    votes: dict[str, int] = {}
    score_sum: dict[str, float] = {}
    hits_all = []

    for qi in range(n_query):
        t_q = float(query_times[qi]) if query_times is not None else None

        for r in range(top_k):
            idx = int(I[qi, r])
            score = float(D_sim[qi, r])

            info = meta.get(str(idx))
            if not info:
                continue

            vid = info.get("video_id", "unknown")
            ts = float(info.get("timestamp", 0.0))
            fname = info.get("frame_name", "")

            votes[vid] = votes.get(vid, 0) + 1
            score_sum[vid] = score_sum.get(vid, 0.0) + score

            hits_all.append(
                {
                    "video_id": vid,
                    "timestamp": ts,
                    "t_query": t_q,
                    "score": score,
                    "faiss_id": idx,
                    "frame_name": fname,
                    "query_frame_i": qi,
                    "rank": r,
                }
            )

    if not votes:
        return empty_result("No valid matches found in metadata.")

    best_video_id = sorted(votes.keys(), key=lambda v: (votes[v], score_sum[v]), reverse=True)[0]

    best_hits = [h for h in hits_all if h["video_id"] == best_video_id]
    best_hits.sort(key=lambda x: x["score"], reverse=True)

    top1_scores = [h["score"] for h in best_hits if h["rank"] == 0]
    if len(top1_scores) == 0:
        base_conf = 0.0
    else:
        top1_scores = np.array(top1_scores, dtype=np.float32)
        mean_score = float(top1_scores.mean())
        strong_ratio = float((top1_scores >= 0.90).mean())
        base_conf = 0.7 * mean_score + 0.3 * strong_ratio

    align_ratio = 0.0
    inliers = best_hits
    best_offset = 0.0

    have_query_times = (
        query_times is not None
        and len(best_hits) > 0
        and all(h["t_query"] is not None for h in best_hits)
    )

    if have_query_times:
        offsets = np.array([h["timestamp"] - h["t_query"] for h in best_hits], dtype=np.float32)

        bins = np.floor(offsets / bin_size).astype(int)
        bin_counts = {}
        for b in bins:
            bin_counts[b] = bin_counts.get(b, 0) + 1

        best_bin = max(bin_counts, key=bin_counts.get)
        best_offset = float((best_bin + 0.5) * bin_size)

        inliers = []
        for h in best_hits:
            off = h["timestamp"] - h["t_query"]
            if abs(off - best_offset) <= inlier_tol:
                inliers.append(h)

        if len(inliers) < max(5, int(0.3 * len(best_hits))):
            inliers = best_hits

        align_ratio = len(inliers) / max(1, len(best_hits))

    confidence = 0.6 * base_conf + 0.4 * align_ratio
    confidence = float(np.clip(confidence, 0.0, 1.0))
    confidence = round(confidence, 4)

    if have_query_times and len(inliers) > 0:
        times_est = np.array([h["t_query"] + best_offset for h in inliers], dtype=np.float32)
    else:
        times_est = np.array([h["timestamp"] for h in inliers], dtype=np.float32)

    scores_est = np.array([h["score"] for h in inliers], dtype=np.float32)

    # est_timestamp: GLOBAL weighted over all inliers
    if len(times_est) > 0:
        w_all = np.maximum(scores_est.astype(np.float64), 1e-6)
        t_all = times_est.astype(np.float64)
        est_ts_global = float((t_all * w_all).sum() / float(w_all.sum()))
        est_ts_global = round(est_ts_global, 2)
    else:
        est_ts_global = None

    # time_window: best segment window, tie-break by score sum
    start_t, end_t = None, None
    if len(times_est) > 0:
        W = float(window_size)
        order = np.argsort(times_est)
        times_sorted = times_est[order]
        scores_sorted = scores_est[order]

        best_i, best_j = 0, 1
        best_count = 1
        best_score_sum = float(scores_sorted[0])

        j = 0
        for i in range(len(times_sorted)):
            while j < len(times_sorted) and times_sorted[j] <= times_sorted[i] + W:
                j += 1

            count = j - i
            score_sum_w = float(scores_sorted[i:j].sum()) if j > i else 0.0

            if (count > best_count) or (count == best_count and score_sum_w > best_score_sum):
                best_i, best_j = i, j
                best_count = count
                best_score_sum = score_sum_w

        cluster_times = times_sorted[best_i:best_j]
        if len(cluster_times) == 0:
            cluster_times = times_sorted

        start_t = round(float(cluster_times[0]), 2)
        end_t = round(float(cluster_times[-1]), 2)

    # Evidence: dedupe
    seen = set()
    top_evidence = []
    for h in best_hits:
        key = (int(h["faiss_id"]), round(float(h["timestamp"]), 2))
        if key in seen:
            continue
        seen.add(key)
        top_evidence.append(h)
        if len(top_evidence) == 10:
            break

    return {
        "best_video_id": best_video_id,
        "confidence": confidence,
        "est_timestamp": est_ts_global,
        "time_window": [start_t, end_t],
        "votes_per_video": votes,
        "score_sum_per_video": {k: round(v, 4) for k, v in score_sum.items()},
        "top_evidence": top_evidence,
        "reason": None,
    }


def search_and_aggregate(
    index,
    meta: dict,
    query_vecs: np.ndarray,
    query_times: np.ndarray | None,
    top_k: int,
    window_size: float = 6.0,
    bin_size: float = 0.5,
    inlier_tol: float = 0.75,
) -> Dict[str, Any]:
    """
    FAISS search + V1 aggregation.
    """
    D, I = index.search(query_vecs.astype(np.float32), int(top_k))
    D_sim = faiss_scores_to_similarity(D, index)
    return aggregate_votes(
        I=I,
        D_sim=D_sim,
        meta=meta,
        query_times=query_times,
        window_size=window_size,
        bin_size=bin_size,
        inlier_tol=inlier_tol,
    )
