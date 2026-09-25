import numpy as np
import pytest

from sceneid.bench.download import require_films
from sceneid.bench.embedding import build_library, embed_queries, load_query_store
from sceneid.bench.queries import build_queries
from sceneid.config import Settings
from sceneid.embedders import TinyImageEmbedder
from tests.conftest import BENCH_DISTORTIONS, needs_ffmpeg

SETTINGS = Settings(embedder="tiny16", index_fps=2.0, batch_size=16)


def test_library_holds_only_library_films_and_resumes(mini_manifest, bench_workspace, tmp_path):
    films = require_films(bench_workspace, mini_manifest)
    embedder = TinyImageEmbedder(16)
    library = build_library(mini_manifest, films, embedder, SETTINGS, tmp_path / "lib")
    assert library.n_videos == 2 and library.get("film-c") is None
    # A second run finds both films already indexed and changes nothing.
    again = build_library(mini_manifest, films, embedder, SETTINGS, tmp_path / "lib")
    assert again.n_vectors == library.n_vectors


@needs_ffmpeg
def test_query_embedding_resumes_without_redoing_work(mini_manifest, bench_workspace, tmp_path):
    films = require_films(bench_workspace, mini_manifest)
    specs = build_queries(mini_manifest, films, seed=0, distortions=BENCH_DISTORTIONS)
    embedder = TinyImageEmbedder(16)
    out = tmp_path / "emb"
    half = len(specs) // 2

    embed_queries(
        specs, films, embedder, SETTINGS, out, queries_digest="d", shard_size=7, limit=half
    )
    first = load_query_store(out)
    assert len(first.queries) == half
    embed_queries(specs, films, embedder, SETTINGS, out, queries_digest="d", shard_size=7)
    store = load_query_store(out)
    assert len(store.queries) + len(store.failed) == len(specs)
    # Queries stored by the first run come back unchanged.
    some_id = next(iter(first.queries))
    assert (store.queries[some_id].vectors == first.queries[some_id].vectors).all()
    assert set(store.queries[some_id].timings_ms) == {"render", "decode", "embed"}


def test_embeddings_from_different_settings_are_never_mixed(
    mini_manifest, bench_workspace, tmp_path
):
    films = require_films(bench_workspace, mini_manifest)
    specs = build_queries(mini_manifest, films, seed=0, distortions=("original",))[:1]
    embedder = TinyImageEmbedder(16)
    embed_queries(specs, films, embedder, SETTINGS, tmp_path, queries_digest="d", limit=0)
    other = Settings(embedder="tiny16", query_fps=1)
    with pytest.raises(ValueError, match="other settings"):
        embed_queries(specs, films, embedder, other, tmp_path, queries_digest="d")


def test_one_pass_fills_every_embedders_library(mini_manifest, bench_workspace, tmp_path):
    from sceneid.bench.embedding import build_libraries
    from sceneid.embedders import PerceptualHashEmbedder

    films = require_films(bench_workspace, mini_manifest)
    tiny, phash = TinyImageEmbedder(16), PerceptualHashEmbedder()
    dirs = {"tiny16": tmp_path / "tiny", "phash64": tmp_path / "phash"}
    # tiny16 already holds everything, so only phash64 needs the films decoded.
    build_library(mini_manifest, films, tiny, SETTINGS, dirs["tiny16"])
    libs = build_libraries(mini_manifest, films, [tiny, phash], SETTINGS, dirs)
    assert libs["tiny16"].n_vectors == libs["phash64"].n_vectors > 0
    assert [v.video_id for v in libs["phash64"].videos()] == ["film-a", "film-b"]


@needs_ffmpeg
def test_rendered_clips_are_cached_and_reused(
    mini_manifest, bench_workspace, tmp_path, monkeypatch
):
    from sceneid.bench import embedding
    from sceneid.embedders import PerceptualHashEmbedder

    films = require_films(bench_workspace, mini_manifest)
    specs = build_queries(mini_manifest, films, seed=0, distortions=("original", "mirror"))[:6]
    clips = tmp_path / "clips"
    embed_queries(
        specs, films, TinyImageEmbedder(16), SETTINGS, tmp_path / "a", queries_digest="d",
        clips_dir=clips,
    )  # fmt: skip
    assert sorted(p.name for p in clips.glob("*.mp4")) == sorted(f"{s.query_id}.mp4" for s in specs)

    # A new embedder reuses every clip: rendering again would fail this test.
    def no_render(*args, **kwargs):
        raise AssertionError("rendered a clip that was cached")

    monkeypatch.setattr(embedding, "render", no_render)
    embed_queries(
        specs, films, PerceptualHashEmbedder(), SETTINGS, tmp_path / "b", queries_digest="d",
        clips_dir=clips,
    )  # fmt: skip
    store = load_query_store(tmp_path / "b")
    assert len(store.queries) == len(specs) and not store.failed
    assert all(np.isnan(q.timings_ms["render"]) for q in store.queries.values())


@needs_ffmpeg
def test_one_pass_embeds_every_query_with_every_embedder(mini_manifest, bench_workspace, tmp_path):
    from sceneid.embedders import PerceptualHashEmbedder

    films = require_films(bench_workspace, mini_manifest)
    specs = build_queries(mini_manifest, films, seed=0, distortions=("original",))
    embedders = [TinyImageEmbedder(16), PerceptualHashEmbedder()]
    dirs = {e.name: tmp_path / e.name for e in embedders}
    result = embed_queries(specs, films, embedders, SETTINGS, dirs, queries_digest="d")
    assert result["processed"] == len(specs)
    a, b = load_query_store(dirs["tiny16"]), load_query_store(dirs["phash64"])
    assert set(a.queries) == set(b.queries) == {s.query_id for s in specs}
    some = specs[0].query_id
    assert (a.queries[some].times == b.queries[some].times).all()  # same decoded frames
    assert a.queries[some].vectors.shape[1] == 256 and b.queries[some].vectors.shape[1] == 64
