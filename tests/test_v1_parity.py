"""The V1 port must reproduce the original V1 code exactly, or the baseline numbers lie."""

import numpy as np
import pytest

from sceneid.library import SearchHits
from sceneid.matching.v1 import V1Voting
from tests.legacy.v1_match_local import aggregate_votes, apply_unknown_gate


def _random_case(rng: np.random.Generator):
    """A random library and a set of top-k search results, in both formats."""
    n_videos = int(rng.integers(1, 6))
    ref_times = [np.arange(0, rng.uniform(5, 60), 0.5, dtype=np.float32) for _ in range(n_videos)]
    frames = [(v, t) for v in range(n_videos) for t in ref_times[v]]
    meta = {
        str(i): {"video_id": f"vid{v}", "timestamp": float(t)} for i, (v, t) in enumerate(frames)
    }

    n_query = int(rng.integers(1, 41))
    k = int(min(rng.integers(1, 11), len(frames)))
    query_times = (np.arange(n_query) / 3.0).astype(np.float32)

    # Half the cases contain a real match: one video at a fixed offset, with noise.
    true_video = int(rng.integers(n_videos))
    offset = float(rng.uniform(0, max(0.1, ref_times[true_video][-1] - query_times[-1])))
    p_true = rng.uniform(0.2, 1.0) if rng.random() < 0.5 else 0.0

    ids = np.zeros((n_query, k), dtype=np.int64)
    for qi in range(n_query):
        row = list(rng.choice(len(frames), size=k, replace=False))
        if rng.random() < p_true:
            target = query_times[qi] + offset + rng.normal(0, 0.3)
            cands = [i for i, (v, _) in enumerate(frames) if v == true_video]
            nearest = min(cands, key=lambda i: abs(frames[i][1] - target))
            if nearest in row:
                row.remove(nearest)
            row = [nearest, *row[: k - 1]]
        ids[qi] = row
    D = -np.sort(-rng.uniform(0.3, 1.0, size=(n_query, k)).astype(np.float32), axis=1)
    if rng.random() < 0.2:  # FAISS returns -1 when it has fewer than k results
        ids[:, -1] = -1

    keys = np.array([[frames[i][0] if i >= 0 else -1 for i in row] for row in ids], dtype=np.int32)
    times = np.array([[frames[i][1] if i >= 0 else 0.0 for i in row] for row in ids], np.float32)
    return ids, D, meta, query_times, SearchHits(D, keys, times)


@pytest.mark.parametrize("seed", range(400))
def test_port_matches_original(seed):
    rng = np.random.default_rng(seed)
    ids, D, meta, query_times, hits = _random_case(rng)

    legacy = aggregate_votes(ids, D, meta, query_times=query_times)
    legacy_gated = apply_unknown_gate(legacy, min_conf=0.83, min_vote_ratio=0.90)
    port = V1Voting(min_conf=0.83, min_vote_ratio=0.90).decide(query_times, hits)

    if legacy["best_video_id"] is None:  # every hit was empty
        assert port.candidate_key is None and not port.accepted
        return
    assert f"vid{port.candidate_key}" == legacy["best_video_id"]
    assert port.confidence == legacy["confidence"]
    assert port.diagnostics["est_timestamp"] == legacy["est_timestamp"]
    assert list(port.window) == legacy["time_window"]
    assert {f"vid{k}": n for k, n in port.votes.items()} == legacy["votes_per_video"]
    assert port.accepted == (legacy_gated["best_video_id"] is not None)
    legacy_refs = [round(h["timestamp"], 2) for h in legacy["top_evidence"]]
    assert [round(ref, 2) for _, ref, _ in port.evidence] == legacy_refs


def test_fallback_gives_full_alignment_to_unaligned_hits():
    """Documents the V1 flaw the port keeps on purpose (see sceneid.matching.v1)."""
    n = 12
    rng = np.random.default_rng(0)
    hits = SearchHits(
        scores=np.full((n, 1), 0.8, np.float32),
        video_keys=np.zeros((n, 1), np.int32),
        ref_times=rng.permutation(np.arange(0, 60, 0.5, dtype=np.float32))[:n].reshape(n, 1),
    )
    decision = V1Voting().decide(np.arange(n, dtype=np.float32) / 3, hits)
    assert decision.diagnostics["aligned"] == 0
    assert decision.diagnostics["align_ratio"] == 1.0
