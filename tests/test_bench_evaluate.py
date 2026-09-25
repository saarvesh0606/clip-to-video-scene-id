import json

import numpy as np
import pytest

from sceneid.bench.download import require_films
from sceneid.bench.embedding import build_library, embed_queries, load_query_store
from sceneid.bench.evaluate import (
    Decisions,
    RunConfig,
    choose_thresholds,
    run_evaluation,
    truth_for,
)
from sceneid.bench.queries import QuerySpec, build_queries
from sceneid.bench.report import write_results
from sceneid.config import Settings
from sceneid.embedders import TinyImageEmbedder
from sceneid.matching import OffsetVoting, V1Voting
from tests.conftest import BENCH_DISTORTIONS, needs_ffmpeg


def _spec(i: int, role: str, split: str) -> QuerySpec:
    return QuerySpec(
        query_id=f"q{i}", base_id=f"b{i}", film_id="film-a" if role == "library" else "film-x",
        role=role, group="g", split=split, ref_start_s=0.0, ref_duration_s=1.0, length_s=1.0,
        distortion="original", seed=0, true_offset_s=0.0,
    )  # fmt: skip


def _synthetic(val_known, val_unknown, test_known=(), test_unknown=()):
    """Queries whose candidate is always film-a, with the given confidences."""
    groups = [
        (val_known, "library", "val"),
        (val_unknown, "heldout", "val"),
        (test_known, "library", "test"),
        (test_unknown, "heldout", "test"),
    ]
    specs, conf = [], []
    for values, role, split in groups:
        for c in values:
            specs.append(_spec(len(specs), role, split))
            conf.append(c)
    n = len(specs)
    decisions = Decisions(
        candidate=np.array(["film-a"] * n, dtype=object),
        offset=np.zeros(n),
        confidence=np.array(conf),
        features={"ratio": np.zeros(n)},
        decide_ms=np.zeros(n),
    )
    return decisions, truth_for(specs, set())


def test_thresholds_are_the_best_dir_within_the_far_target():
    known = np.linspace(0.80, 0.99, 10)
    unknown = [0.95, 0.85, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1]
    decisions, truth = _synthetic(known, unknown)
    alg = OffsetVoting()

    thresholds, selection = choose_thresholds(alg, decisions, truth, target_far=0.1)
    assert selection["val_target_met"]
    # Just above the 0.85 unknown: one false accept of ten (10%), as many known clips as possible.
    assert 0.85 < thresholds["min_score"] <= 0.86
    assert thresholds["max_ratio"] == 0.5  # ties go to the strictest ratio

    strict, _ = choose_thresholds(alg, decisions, truth, target_far=0.0)
    assert strict["min_score"] > 0.95


def test_test_split_never_influences_the_thresholds():
    known = np.linspace(0.80, 0.99, 10)
    unknown = [0.95, 0.85, 0.6, 0.5, 0.4]
    a, truth_a = _synthetic(known, unknown)
    b, truth_b = _synthetic(known, unknown, test_known=[0.1] * 5, test_unknown=[0.99] * 5)
    alg = OffsetVoting()
    assert choose_thresholds(alg, a, truth_a, 0.2) == choose_thresholds(alg, b, truth_b, 0.2)


@pytest.fixture(scope="module")
def embedded(mini_manifest, bench_workspace, tmp_path_factory):
    films = require_films(bench_workspace, mini_manifest)
    specs = build_queries(mini_manifest, films, seed=0, distortions=BENCH_DISTORTIONS)
    settings = Settings(embedder="tiny16", index_fps=2.0, batch_size=16)
    embedder = TinyImageEmbedder(16)
    root = tmp_path_factory.mktemp("evaluate")
    library = build_library(mini_manifest, films, embedder, settings, root / "lib")
    embed_queries(specs, films, embedder, settings, root / "emb", queries_digest="d")
    return library, specs, load_query_store(root / "emb")


@needs_ffmpeg
def test_evaluation_and_report(embedded, mini_manifest, tmp_path):
    library, specs, store = embedded
    runs = [
        RunConfig("v1-shipped", V1Voting(), tune=False),
        RunConfig("v1-tuned", V1Voting(), tune=True),
        RunConfig("v2-tuned", OffsetVoting(), tune=True),
    ]
    summary, results, rows = run_evaluation(
        library, specs, store, runs, top_k=10, target_far=0.1, bootstrap=100
    )
    assert summary["queries"]["n_queries"] == len(specs) == len(rows)
    shipped = next(r for r in results if r["name"] == "v1-shipped")
    assert shipped["thresholds"] == {"min_conf": 0.83, "min_vote_ratio": 0.9}

    v2 = next(r for r in results if r["name"] == "v2-tuned")
    assert v2["by_distortion"]["original"]["top1"] == 1.0
    assert v2["by_distortion"]["original"]["offset_within_1s"] == 1.0
    low, high = v2["test"]["ci95"]["dir"]
    assert low <= v2["test"]["dir"] <= high
    assert list(v2["by_distortion"]) == list(BENCH_DISTORTIONS)  # defined order, not alphabetical

    out = write_results(
        tmp_path / "results",
        {
            **summary,
            "created_at": "2026-01-01T00:00:00+00:00",
            "sceneid": "test",
            "git": None,
            "manifest": {"name": "mini", "digest": mini_manifest.digest, "n_heldout": 1},
            "embeddings": store.meta,
            "evaluated_on": {"platform": "test"},
        },
        results,
        rows,
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Headline" in report and "v2-tuned" in report
    assert json.loads((out / "summary.json").read_text())["runs"][0]["name"] == "v1-shipped"
    assert (out / "per_query.csv").read_text().count("\n") == len(specs) + 1


@needs_ffmpeg
def test_partial_embeddings_are_refused_unless_allowed(embedded):
    library, specs, store = embedded
    extra = [*specs, _spec(10_000, "library", "test")]
    runs = [RunConfig("v2", OffsetVoting(), tune=False)]
    with pytest.raises(ValueError, match="not embedded yet"):
        run_evaluation(library, extra, store, runs, top_k=10, bootstrap=10)
    summary, _, _ = run_evaluation(
        library, extra, store, runs, top_k=10, bootstrap=10, allow_partial=True
    )
    assert summary["queries"]["n_missing"] == 1
