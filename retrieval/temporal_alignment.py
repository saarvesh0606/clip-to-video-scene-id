# retrieval/temporal_alignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class TemporalResult:
    est_offset: float | None
    est_timestamp: float | None
    time_window: Tuple[float | None, float | None]
    alignment_ratio: float
    cluster_count: int


def _cluster_offsets(offsets: List[float], bin_size_sec: float = 1.0) -> Tuple[float | None, int, float]:
    """
    Simple 1D histogram clustering for offsets.
    Returns (best_offset_center, best_count, best_ratio).
    """
    if not offsets:
        return None, 0, 0.0

    arr = np.array(offsets, dtype=np.float32)
    # Bin offsets
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return float(arr.mean()), len(offsets), 1.0

    bins = np.arange(min_v, max_v + bin_size_sec, bin_size_sec, dtype=np.float32)
    hist, edges = np.histogram(arr, bins=bins)

    best_idx = int(hist.argmax())
    best_count = int(hist[best_idx])
    total = int(hist.sum())
    best_ratio = float(best_count / max(total, 1))

    # Use center of the winning bin
    left = float(edges[best_idx])
    right = float(edges[best_idx + 1])
    center = (left + right) / 2.0
    return center, best_count, best_ratio


def estimate_timestamp_from_matches(
    per_frame_best_ref_time: Dict[int, float],
    per_frame_query_time: Dict[int, float],
    window_sec: float = 2.0,
    bin_size_sec: float = 1.0
) -> TemporalResult:
    """
    Args:
      per_frame_best_ref_time: map q_frame_idx -> matched ref timestamp (seconds)
      per_frame_query_time:    map q_frame_idx -> query timestamp (seconds)
      window_sec: +/- around estimated timestamp for output window
      bin_size_sec: clustering granularity

    Returns:
      TemporalResult
    """
    offsets: List[float] = []
    for qi, tr in per_frame_best_ref_time.items():
        tq = per_frame_query_time.get(qi, None)
        if tq is None:
            continue
        offsets.append(float(tr - tq))

    best_offset, best_count, align_ratio = _cluster_offsets(offsets, bin_size_sec=bin_size_sec)
    if best_offset is None:
        return TemporalResult(
            est_offset=None,
            est_timestamp=None,
            time_window=(None, None),
            alignment_ratio=0.0,
            cluster_count=0,
        )

    # Use the median query time as anchor, estimate timestamp in ref:
    q_times = [per_frame_query_time[i] for i in per_frame_best_ref_time.keys() if i in per_frame_query_time]
    if not q_times:
        return TemporalResult(
            est_offset=float(best_offset),
            est_timestamp=None,
            time_window=(None, None),
            alignment_ratio=float(align_ratio),
            cluster_count=int(best_count),
        )

    anchor_q = float(np.median(np.array(q_times, dtype=np.float32)))
    est_ts = float(anchor_q + best_offset)
    return TemporalResult(
        est_offset=float(best_offset),
        est_timestamp=est_ts,
        time_window=(est_ts - window_sec, est_ts + window_sec),
        alignment_ratio=float(align_ratio),
        cluster_count=int(best_count),
    )
