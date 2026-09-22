from pathlib import Path

import pytest

from sceneid.embedders import TinyImageEmbedder
from sceneid.indexer import default_video_id, index_video
from sceneid.library import LibraryError
from sceneid.matcher import Matcher
from sceneid.matching import V1Voting
from tests.conftest import INDEX_FPS


def test_index_records_the_video(library):
    record = library.get("ref_a")
    assert record.n_vectors == 60
    assert record.duration_s == pytest.approx(30.0, abs=0.1)
    assert len(record.sha256) == 64


def test_reindexing_needs_replace(library, embedder, videos):
    kwargs = dict(fps=INDEX_FPS, short_side=None, batch_size=16)
    with pytest.raises(LibraryError, match="--replace"):
        index_video(library, embedder, videos["ref_a"], **kwargs)
    index_video(library, embedder, videos["ref_a"], replace=True, **kwargs)
    assert library.n_videos == 2 and library.n_vectors == 120


def test_same_file_under_a_new_id_is_refused(library, embedder, videos):
    with pytest.raises(LibraryError, match="already indexed as 'ref_a'"):
        index_video(
            library,
            embedder,
            videos["ref_a"],
            video_id="copy",
            fps=2,
            short_side=None,
            batch_size=8,
        )


def test_default_video_id_is_sanitised():
    assert default_video_id(Path("My Movie (2020).mp4")) == "My-Movie-2020"
    assert default_video_id(Path("???.mp4")) == "video"


def test_clip_is_matched_to_its_source_and_start_time(matcher, videos):
    result = matcher.identify(videos["clip_a"])
    assert result.status == "match"
    assert result.video_id == "ref_a"
    assert result.offset_s == pytest.approx(11.0, abs=0.5)
    assert result.n_query_frames == 15
    assert set(result.timings_ms) >= {"decode", "embed", "search", "decide", "total"}


def test_clip_from_an_unindexed_video_is_unknown(matcher, videos):
    result = matcher.identify(videos["unknown"])
    assert result.status == "unknown"
    assert result.video_id is None and result.offset_s is None
    assert result.candidate.video_id in {"ref_a", "ref_b"}


def test_matcher_refuses_an_embedder_the_library_was_not_built_with(library):
    with pytest.raises(ValueError, match="built with tiny16"):
        Matcher(
            TinyImageEmbedder(8),
            library,
            V1Voting(),
            query_fps=3,
            max_frames=40,
            top_k=10,
            short_side=None,
        )
