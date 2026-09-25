import numpy as np
import pytest

from sceneid.library import SearchHits
from sceneid.matcher import Matcher
from sceneid.matching import OffsetVoting, V1Voting


def _hits(rows: list[list[tuple[int, float, float]]]) -> SearchHits:
    """Build SearchHits from per-query-frame lists of (video_key, ref_time, score)."""
    k = max(len(r) for r in rows)
    keys = np.full((len(rows), k), -1, np.int32)
    times = np.zeros((len(rows), k), np.float32)
    scores = np.zeros((len(rows), k), np.float32)
    for i, row in enumerate(rows):
        for j, (key, t, s) in enumerate(row):
            keys[i, j], times[i, j], scores[i, j] = key, t, s
    return SearchHits(scores, keys, times)


QUERY_TIMES = np.arange(12, dtype=np.float32) / 3  # 12 frames at 3 fps


def test_consistent_offset_is_accepted_with_a_precise_timestamp():
    rng = np.random.default_rng(0)
    rows = []
    for t in QUERY_TIMES:
        true_ref = t + 7.3 + rng.normal(0, 0.1)
        # The true frame plus distractors from the same video and another one.
        rows.append([(0, true_ref, 0.95), (0, true_ref + 4, 0.80), (1, rng.uniform(0, 60), 0.75)])
    d = OffsetVoting().decide(QUERY_TIMES, _hits(rows))
    assert d.accepted and d.candidate_key == 0
    assert d.offset_s == pytest.approx(7.3, abs=0.1)
    assert d.confidence == pytest.approx(0.95, abs=0.01)
    assert d.window[0] == pytest.approx(7.3, abs=0.1)


def test_hits_that_do_not_line_up_are_rejected():
    """The case V1's fallback scores as perfectly aligned (see test_v1_parity)."""
    rng = np.random.default_rng(0)
    ref_times = rng.permutation(np.arange(0, 60, 0.5))[: len(QUERY_TIMES)]
    hits = _hits([[(0, t, 0.8)] for t in ref_times])

    v1 = V1Voting().decide(QUERY_TIMES, hits)
    v2 = OffsetVoting().decide(QUERY_TIMES, hits)
    assert v1.diagnostics["align_ratio"] == 1.0
    assert not v2.accepted and v2.reason == "low_score"
    assert v2.confidence < 0.3


def test_neighbours_from_one_static_shot_count_once_per_frame():
    # Every query frame's ten neighbours are one shot, 5 s long.
    rows = [[(0, 20 + j * 0.5, 0.9) for j in range(10)] for _ in QUERY_TIMES]
    d = OffsetVoting().decide(QUERY_TIMES, _hits(rows))
    assert d.confidence <= 0.9
    assert d.votes[0] <= len(QUERY_TIMES)


def test_a_clip_that_fits_two_videos_equally_is_ambiguous():
    rows = [[(0, t + 5, 0.93), (1, t + 40, 0.92)] for t in QUERY_TIMES]
    d = OffsetVoting().decide(QUERY_TIMES, _hits(rows))
    assert not d.accepted and d.reason == "ambiguous"
    assert d.candidate_key == 0
    assert d.diagnostics["ratio"] > 0.9


def test_no_candidates():
    empty = SearchHits(
        np.zeros((3, 2), np.float32), np.full((3, 2), -1, np.int32), np.zeros((3, 2), np.float32)
    )
    d = OffsetVoting().decide(QUERY_TIMES[:3], empty)
    assert not d.accepted and d.candidate_key is None and d.reason == "no_candidates"


@pytest.fixture
def v2_matcher(library, embedder) -> Matcher:
    return Matcher(
        embedder,
        library,
        OffsetVoting(),
        query_fps=3.0,
        max_frames=40,
        top_k=10,
        short_side=None,
    )


def test_end_to_end_match_and_unknown(v2_matcher, videos):
    match = v2_matcher.identify(videos["clip_a"])
    assert match.status == "match" and match.video_id == "ref_a"
    assert match.offset_s == pytest.approx(11.0, abs=0.2)
    unknown = v2_matcher.identify(videos["unknown"])
    assert unknown.status == "unknown" and unknown.reason == "low_score"


def test_long_clips_are_sampled_across_their_whole_length(v2_matcher, videos):
    v2_matcher.max_frames = 10
    q = v2_matcher.embed_clip(videos["ref_a"])  # 30 s
    assert len(q.times) == 10
    assert q.times[-1] > 25
