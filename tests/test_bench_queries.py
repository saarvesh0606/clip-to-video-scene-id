import pytest

from sceneid.bench.distortions import DISTORTIONS, render
from sceneid.bench.download import require_films
from sceneid.bench.queries import build_queries, load_queries, save_queries
from sceneid.frames import probe
from tests.conftest import BENCH_DISTORTIONS, needs_ffmpeg


@pytest.fixture(scope="module")
def films(mini_manifest, bench_workspace):
    return require_films(bench_workspace, mini_manifest)


def test_answer_key_is_deterministic(mini_manifest, films):
    specs = build_queries(mini_manifest, films, seed=3, distortions=BENCH_DISTORTIONS)
    assert specs == build_queries(mini_manifest, films, seed=3, distortions=BENCH_DISTORTIONS)
    assert specs != build_queries(mini_manifest, films, seed=4, distortions=BENCH_DISTORTIONS)


def test_every_base_clip_is_rendered_each_way_in_one_split(mini_manifest, films):
    specs = build_queries(mini_manifest, films, seed=3, distortions=BENCH_DISTORTIONS)
    by_base: dict[str, list] = {}
    for s in specs:
        by_base.setdefault(s.base_id, []).append(s)
    assert all(len(v) == len(BENCH_DISTORTIONS) for v in by_base.values())
    assert all(len({s.split for s in v}) == 1 for v in by_base.values())
    assert {s.split for s in specs} == {"val", "test"}
    assert {s.expected_video_id for s in specs if s.role == "heldout"} == {None}


def test_clips_skip_titles_and_credits_and_never_overlap(mini_manifest, films):
    specs = build_queries(mini_manifest, films, seed=3, distortions=BENCH_DISTORTIONS)
    for film_id in ("film-a", "film-b", "film-c"):
        spans = sorted(
            (s.ref_start_s, s.ref_start_s + s.length_s * 1.25)
            for s in specs
            if s.film_id == film_id and s.distortion == "original"
        )
        assert len(spans) == 4  # min_clips for a 90-second film
        assert spans[0][0] >= 90 * 0.08 - 0.01 and spans[-1][1] <= 90 * 0.92 + 0.01
        assert all(end < start for (_, end), (start, _) in zip(spans, spans[1:], strict=False))


def test_speed_ups_cover_more_of_the_film(mini_manifest, films):
    specs = build_queries(mini_manifest, films, seed=3, distortions=BENCH_DISTORTIONS)
    speed = next(s for s in specs if s.distortion == "speed")
    assert speed.ref_duration_s == pytest.approx(speed.length_s * 1.25, abs=0.01)


def test_answer_key_round_trips(mini_manifest, films, tmp_path):
    specs = build_queries(mini_manifest, films, seed=0, distortions=BENCH_DISTORTIONS)
    save_queries(specs, tmp_path / "q.jsonl", {"seed": 0})
    loaded, meta = load_queries(tmp_path / "q.jsonl")
    assert loaded == specs and meta == {"seed": 0}


def test_unknown_distortions_are_refused(mini_manifest, films):
    with pytest.raises(ValueError, match="unknown distortions"):
        build_queries(mini_manifest, films, distortions=("original", "sepia"))


@needs_ffmpeg
@pytest.mark.parametrize("name", list(DISTORTIONS))
def test_every_distortion_renders_a_readable_clip(name, tmp_path, videos):
    out = tmp_path / f"{name}.mp4"
    speed = DISTORTIONS[name].speed
    render(name, videos["ref_a"], 5.0, 4.0 * speed, out, seed=1, width=160, height=96)
    info = probe(out)
    assert info.duration_s == pytest.approx(4.0, abs=0.25)
    assert info.width % 2 == 0 and info.height % 2 == 0
