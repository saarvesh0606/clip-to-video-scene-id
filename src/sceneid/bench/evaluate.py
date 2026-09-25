"""Scoring a benchmark: search, decide, choose thresholds on val, report on test.

Terms, following open-set identification practice:

* **known** clip: cut from an indexed film. **unknown** clip: cut from a held-out film.
* **DIR** (detection and identification rate): known clips accepted *and* matched to the
  right film. This is the headline accuracy.
* **FAR** (false accept rate): unknown clips accepted as some film.
* **FRR**: known clips rejected. **misID**: known clips accepted as the wrong film.
* **top-1**: known clips whose best candidate is right, ignoring the accept/reject gate.

Thresholds are chosen on the val split as the ones with the highest DIR whose FAR stays at
or under the target, then frozen and applied to the test split. Confidence intervals come
from resampling *base clips* (all distortions of one clip move together), because those
renderings are not independent.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..library import Library
from ..matcher import QueryEmbedding
from ..matching import Algorithm
from .embedding import QueryStore
from .queries import QuerySpec

log = logging.getLogger(__name__)

GRIDS: dict[str, dict[str, list[float]]] = {
    "v1": {
        "min_conf": [round(x, 3) for x in np.linspace(0, 1, 201)],
        "min_vote_ratio": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    },
    "v2": {
        "min_score": [round(x, 3) for x in np.linspace(0, 1, 201)],
        "max_ratio": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    },
}


@dataclass
class RunConfig:
    name: str  # e.g. "v1-shipped"
    algorithm: Algorithm
    tune: bool  # choose thresholds on val, or keep the algorithm's own


@dataclass
class Decisions:
    """One algorithm's answer for every query, as arrays aligned with the query list."""

    candidate: np.ndarray  # object: film id or None
    offset: np.ndarray  # float, nan when there is no candidate
    confidence: np.ndarray
    features: dict[str, np.ndarray]  # diagnostics the gate reads
    decide_ms: np.ndarray


@dataclass
class Truth:
    specs: list[QuerySpec]
    known: np.ndarray
    split: np.ndarray
    base_index: np.ndarray  # which base clip each query came from, for the bootstrap
    failed: np.ndarray  # rendering failed: counted as a rejection
    distortion: np.ndarray
    length: np.ndarray
    film: np.ndarray
    group: np.ndarray
    true_offset: np.ndarray
    n_base: int = 0


def truth_for(specs: list[QuerySpec], failed: set[str]) -> Truth:
    bases = sorted({s.base_id for s in specs})
    base_pos = {b: i for i, b in enumerate(bases)}
    arr = lambda f, dtype=object: np.array([f(s) for s in specs], dtype=dtype)  # noqa: E731
    return Truth(
        specs=specs,
        known=arr(lambda s: s.role == "library", bool),
        split=arr(lambda s: s.split),
        base_index=arr(lambda s: base_pos[s.base_id], np.int64),
        failed=arr(lambda s: s.query_id in failed, bool),
        distortion=arr(lambda s: s.distortion),
        length=arr(lambda s: s.length_s, float),
        film=arr(lambda s: s.film_id),
        group=arr(lambda s: s.group),
        true_offset=arr(lambda s: s.true_offset_s, float),
        n_base=len(bases),
    )


def search_all(
    library: Library, specs: list[QuerySpec], store: QueryStore, k: int, chunk: int = 8192
):
    """Nearest reference frames for every query, in one batched pass."""
    present = [s for s in specs if s.query_id in store.queries]
    queries = [store.queries[s.query_id] for s in present]
    if not queries:
        raise ValueError("no embedded queries to evaluate")
    vectors = np.concatenate([q.vectors for q in queries])
    parts = [library.search(vectors[i : i + chunk], k) for i in range(0, len(vectors), chunk)]
    scores = np.concatenate([p.scores for p in parts])
    keys = np.concatenate([p.video_keys for p in parts])
    times = np.concatenate([p.ref_times for p in parts])
    out, start = {}, 0
    for spec, q in zip(present, queries, strict=True):
        n = len(q.times)
        out[spec.query_id] = (
            scores[start : start + n],
            keys[start : start + n],
            times[start : start + n],
        )
        start += n
    return out


def decide_all(
    algorithm: Algorithm,
    top_k: int,
    library: Library,
    specs: list[QuerySpec],
    store: QueryStore,
    hits: dict,
) -> Decisions:
    from ..library import SearchHits

    n = len(specs)
    candidate = np.full(n, None, dtype=object)
    offset = np.full(n, np.nan)
    confidence = np.zeros(n)
    decide_ms = np.full(n, np.nan)
    features: dict[str, np.ndarray] = {}
    for i, spec in enumerate(specs):
        q: QueryEmbedding | None = store.queries.get(spec.query_id)
        if q is None:
            continue
        s, k, t = hits[spec.query_id]
        started = time.perf_counter()
        d = algorithm.decide(q.times, SearchHits(s[:, :top_k], k[:, :top_k], t[:, :top_k]))
        decide_ms[i] = (time.perf_counter() - started) * 1e3
        if d.candidate_key is None:
            continue
        candidate[i] = library.video_id(d.candidate_key)
        offset[i] = d.offset_s if d.offset_s is not None else np.nan
        confidence[i] = d.confidence
        for name, value in d.diagnostics.items():
            if isinstance(value, (int, float)):
                features.setdefault(name, np.full(n, np.nan))[i] = value
    return Decisions(candidate, offset, confidence, features, decide_ms)


def _gate(algorithm: Algorithm, decisions: Decisions, truth: Truth, thresholds: dict) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        accept = algorithm.gate(decisions.confidence, decisions.features, **thresholds)
    accept = np.asarray(accept, dtype=bool) & np.array([c is not None for c in decisions.candidate])
    return accept & ~truth.failed


def _correct(decisions: Decisions, truth: Truth) -> np.ndarray:
    return (
        np.array([c == f for c, f in zip(decisions.candidate, truth.film, strict=True)])
        & truth.known
    )


def choose_thresholds(
    algorithm: Algorithm, decisions: Decisions, truth: Truth, target_far: float
) -> tuple[dict, dict]:
    """Highest val DIR with val FAR <= target; if nothing reaches the target, lowest FAR."""
    grid = GRIDS[algorithm.name]
    primary, secondary = algorithm.gate_params
    val = truth.split == "val"
    known, unknown = val & truth.known, val & ~truth.known
    correct = _correct(decisions, truth)
    # A "max_" threshold is stricter when lower; a "min_" one when higher.
    strict = -1.0 if secondary.startswith("max_") else 1.0
    best, best_key = None, None
    for b in grid[secondary]:
        for a in grid[primary]:
            thresholds = {primary: a, secondary: b}
            accept = _gate(algorithm, decisions, truth, thresholds)
            dir_ = (accept & correct)[known].mean() if known.any() else 0.0
            far = accept[unknown].mean() if unknown.any() else 0.0
            feasible = far <= target_far + 1e-12
            # Feasible first; then higher DIR (lower FAR if infeasible); ties go to stricter.
            key = (feasible, dir_ if feasible else -far, -far, a, strict * b)
            if best_key is None or key > best_key:
                best, best_key = thresholds, key
    return best, {"val_target_met": bool(best_key[0]), "target_far": target_far}


def _bootstrap_ci(values: np.ndarray, mask: np.ndarray, truth: Truth, weights: np.ndarray) -> list:
    """95% interval of the mean of `values` over `mask`, resampling base clips."""
    w = weights[:, truth.base_index] * mask[None, :]
    denom = w.sum(axis=1)
    ok = denom > 0
    if not ok.any():
        return [None, None]
    stats = (w[ok] @ values.astype(float)) / denom[ok]
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def split_metrics(
    accept, correct, offset_err, truth: Truth, mask: np.ndarray, weights=None
) -> dict:
    known, unknown = mask & truth.known, mask & ~truth.known
    good = accept & correct
    m: dict = {"n_known": int(known.sum()), "n_unknown": int(unknown.sum())}
    if known.any():
        m["dir"] = round(float(good[known].mean()), 4)
        m["frr"] = round(float((~accept)[known].mean()), 4)
        m["misid"] = round(float((accept & ~correct)[known].mean()), 4)
        m["top1"] = round(float(correct[known].mean()), 4)
        errs = offset_err[good & known]
        if len(errs):
            m["offset_median_s"] = round(float(np.median(errs)), 3)
            m["offset_p90_s"] = round(float(np.percentile(errs, 90)), 3)
            m["offset_within_0.5s"] = round(float((errs <= 0.5).mean()), 4)
            m["offset_within_1s"] = round(float((errs <= 1.0).mean()), 4)
    if unknown.any():
        m["far"] = round(float(accept[unknown].mean()), 4)
    if weights is not None:
        ci = {}
        if known.any():
            ci["dir"] = _bootstrap_ci(good, known, truth, weights)
            ci["top1"] = _bootstrap_ci(correct, known, truth, weights)
            within = np.where(good, offset_err <= 1.0, False)
            ci["offset_within_1s"] = _bootstrap_ci(within, good & known, truth, weights)
        if unknown.any():
            ci["far"] = _bootstrap_ci(accept, unknown, truth, weights)
        m["ci95"] = ci
    return m


def _order(value):
    """Distortions in their defined order, numbers numerically, everything else by name."""
    from .distortions import DISTORTIONS

    names = list(DISTORTIONS)
    if value in names:
        return (0, names.index(value), "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0, str(value))


def _breakdown(key: np.ndarray, accept, correct, offset_err, truth: Truth, mask) -> dict:
    out = {}
    for value in sorted({v for v in key[mask]}, key=_order):
        sub = mask & (key == value)
        m = split_metrics(accept, correct, offset_err, truth, sub)
        out[str(value)] = m
    return out


def evaluate_run(
    run: RunConfig,
    decisions: Decisions,
    truth: Truth,
    *,
    target_far: float,
    bootstrap: int,
    seed: int,
) -> dict:
    algorithm = run.algorithm
    if run.tune:
        thresholds, selection = choose_thresholds(algorithm, decisions, truth, target_far)
    else:
        thresholds = {p: getattr(algorithm, p) for p in algorithm.gate_params}
        selection = {"val_target_met": None, "target_far": None}
    accept = _gate(algorithm, decisions, truth, thresholds)
    correct = _correct(decisions, truth)
    offset_err = np.abs(decisions.offset - truth.true_offset)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(truth.n_base, np.full(truth.n_base, 1 / truth.n_base), size=bootstrap)
    test, val = truth.split == "test", truth.split == "val"

    # ROC on test: sweep the primary threshold with the secondary fixed at its chosen value.
    primary, secondary = algorithm.gate_params
    roc = []
    for a in GRIDS[algorithm.name][primary]:
        acc = _gate(algorithm, decisions, truth, {primary: a, secondary: thresholds[secondary]})
        k, u = test & truth.known, test & ~truth.known
        roc.append(
            {
                primary: a,
                "far": round(float(acc[u].mean()), 4) if u.any() else None,
                "dir": round(float((acc & correct)[k].mean()), 4) if k.any() else None,
            }
        )
    return {
        "name": run.name,
        "algorithm": algorithm.name,
        "thresholds": thresholds,
        "selection": selection,
        "test": split_metrics(accept, correct, offset_err, truth, test, weights),
        "val": split_metrics(accept, correct, offset_err, truth, val),
        "by_distortion": _breakdown(truth.distortion, accept, correct, offset_err, truth, test),
        "by_length": _breakdown(truth.length, accept, correct, offset_err, truth, test),
        "by_film": _breakdown(truth.film, accept, correct, offset_err, truth, test),
        "unknown_by_group": _breakdown(
            truth.group, accept, correct, offset_err, truth, test & ~truth.known
        ),
        "roc_test": roc,
        "decide_ms_p50": round(float(np.nanpercentile(decisions.decide_ms, 50)), 3),
        "decide_ms_p95": round(float(np.nanpercentile(decisions.decide_ms, 95)), 3),
        "_accept": accept,
        "_correct": correct,
        "_offset_err": offset_err,
    }


def search_latency(
    library: Library, store: QueryStore, specs: list[QuerySpec], k: int, n: int = 300
):
    """Per-query search time, one query at a time as the API does it (not batched)."""
    sample = [store.queries[s.query_id] for s in specs if s.query_id in store.queries][:n]
    times = []
    for q in sample:
        started = time.perf_counter()
        library.search(q.vectors, k)
        times.append((time.perf_counter() - started) * 1e3)
    return {
        "n": len(times),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }


def _p50_p95(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(values, 50)), 1), round(float(np.percentile(values, 95)), 1)]


def run_evaluation(
    library: Library,
    specs: list[QuerySpec],
    store: QueryStore,
    runs: list[RunConfig],
    *,
    top_k: int,
    target_far: float = 0.01,
    bootstrap: int = 1000,
    seed: int = 0,
    allow_partial: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Evaluate every run. Returns (summary, run results, per-query rows)."""
    if store.meta.get("embedder") != library.embedder:
        raise ValueError(
            f"queries were embedded with {store.meta.get('embedder')}, "
            f"but the library uses {library.embedder}"
        )
    missing = [
        s for s in specs if s.query_id not in store.queries and s.query_id not in store.failed
    ]
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)} of {len(specs)} queries are not embedded yet; finish `bench embed` "
            "or pass --allow-partial"
        )
    missing_ids = {s.query_id for s in missing}
    specs = [s for s in specs if s.query_id not in missing_ids]
    truth = truth_for(specs, set(store.failed))
    hits = search_all(library, specs, store, top_k)

    results = []
    for run in runs:
        log.info("bench.evaluate_run", extra={"run": run.name})
        decisions = decide_all(run.algorithm, top_k, library, specs, store, hits)
        results.append(
            evaluate_run(
                run, decisions, truth, target_far=target_far, bootstrap=bootstrap, seed=seed
            )
        )

    embedded = [store.queries[s.query_id] for s in specs if s.query_id in store.queries]
    latency = {
        "decode_ms": _p50_p95([q.timings_ms["decode"] for q in embedded]),
        "embed_ms": _p50_p95([q.timings_ms["embed"] for q in embedded]),
        "render_ms": _p50_p95([q.timings_ms["render"] for q in embedded]),
        "search": search_latency(library, store, specs, top_k),
    }

    rows = []
    for i, spec in enumerate(specs):
        row = {
            "query_id": spec.query_id,
            "split": spec.split,
            "film": spec.film_id,
            "role": spec.role,
            "group": spec.group,
            "distortion": spec.distortion,
            "length_s": spec.length_s,
            "true_offset_s": spec.true_offset_s,
            "failed": bool(truth.failed[i]),
        }
        for r in results:
            err = r["_offset_err"][i]
            row[f"{r['name']}_accepted"] = bool(r["_accept"][i])
            row[f"{r['name']}_correct"] = bool(r["_correct"][i]) if spec.role == "library" else ""
            row[f"{r['name']}_offset_err_s"] = round(float(err), 3) if np.isfinite(err) else ""
        rows.append(row)

    videos = library.videos()
    summary = {
        "library": {
            "embedder": library.embedder,
            "dim": library.dim,
            "n_videos": library.n_videos,
            "n_vectors": library.n_vectors,
            "hours": round(sum(v.duration_s for v in videos) / 3600, 3),
            "index_fps": videos[0].sample_fps if videos else None,
            "index_mb": round(library.index_bytes / 1e6, 1),
        },
        "queries": {
            "n_queries": len(specs),
            "n_base": truth.n_base,
            "n_distortions": len(set(truth.distortion)),
            "n_known": int(truth.known.sum()),
            "n_unknown": int((~truth.known).sum()),
            "n_failed": int(truth.failed.sum()),
            "n_missing": len(missing),
        },
        "settings": {
            "top_k": top_k,
            "target_far": target_far,
            "bootstrap": bootstrap,
            "seed": seed,
        },
        "latency": latency,
    }
    return summary, results, rows
