"""``sceneid bench``: download films, build the answer key, embed, evaluate.

Typical run (the embed steps want a GPU; see notebooks/benchmark_colab.ipynb):

    sceneid bench download
    sceneid bench queries
    sceneid bench index
    sceneid bench embed
    sceneid bench evaluate
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..config import Settings
from ..embedders import create_embedder
from ..library import Library
from ..matching import OffsetVoting, V1Voting
from .distortions import DISTORTIONS, render
from .download import download_all, load_film_file, require_films
from .embedding import build_library, embed_queries, load_query_store, machine_info
from .evaluate import RunConfig, run_evaluation
from .manifest import load_manifest
from .queries import build_queries, load_queries, save_queries
from .report import git_revision, write_results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sceneid bench", description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, default=Path("benchmarks/datasets/tier1.json"))
    p.add_argument(
        "--workspace", type=Path, default=Path("data/bench"), help="films and working files"
    )
    p.add_argument("--queries", type=Path, help="answer key (default: <workspace>/queries.jsonl)")
    p.add_argument("--library-dir", type=Path, help="default: <workspace>/libraries/<embedder>")
    p.add_argument("--embeddings-dir", type=Path, help="default: <workspace>/embeddings/<embedder>")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download", help="download and verify the films")
    s.add_argument("--only", nargs="+", metavar="FILM_ID")

    s = sub.add_parser("queries", help="build the answer key")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--clips-per-minute", type=float, default=1.0)
    s.add_argument("--distortions", nargs="+", choices=list(DISTORTIONS), default=list(DISTORTIONS))

    sub.add_parser("index", help="index the library films (resumable)")

    s = sub.add_parser("embed", help="render and embed every query clip (resumable)")
    s.add_argument("--workers", type=int, default=2, help="parallel ffmpeg renders")
    s.add_argument("--shard-size", type=int, default=200)
    s.add_argument("--limit", type=int, help="stop after this many queries (for a trial run)")

    s = sub.add_parser("evaluate", help="score the run and write the report")
    s.add_argument("--out", type=Path, help="default: benchmarks/results/<manifest>-<embedder>")
    s.add_argument("--target-far", type=float, default=0.01)
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--allow-partial", action="store_true", help="evaluate what is embedded so far")

    s = sub.add_parser("render", help="write one query clip to a file, to look at it")
    s.add_argument("query_id")
    s.add_argument("--out", type=Path, required=True)

    sub.add_parser("status", help="show what is done so far")
    return p


def _paths(args, settings: Settings):
    queries = args.queries or args.workspace / "queries.jsonl"
    library = args.library_dir or args.workspace / "libraries" / settings.embedder
    embeddings = args.embeddings_dir or args.workspace / "embeddings" / settings.embedder
    return queries, library, embeddings


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str], settings: Settings) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    queries_path, library_dir, embeddings_dir = _paths(args, settings)

    if args.command == "download":
        for film_id, info in download_all(manifest, args.workspace, args.only).items():
            check = "md5 verified" if info.md5_verified else "size verified"
            size = f"{info.width}x{info.height}"
            print(f"{film_id:<28} {info.duration_s / 60:6.1f} min  {size:<9}  {check}")
        return 0

    if args.command == "queries":
        films = require_films(args.workspace, manifest)
        specs = build_queries(
            manifest,
            films,
            seed=args.seed,
            distortions=tuple(args.distortions),
            clips_per_minute=args.clips_per_minute,
        )
        meta = {
            "manifest": manifest.name,
            "manifest_digest": manifest.digest,
            "seed": args.seed,
            "clips_per_minute": args.clips_per_minute,
            "distortions": args.distortions,
            "films_sha256": {k: v.sha256 for k, v in films.items()},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_queries(specs, queries_path, meta)
        bases = {s.base_id for s in specs}
        known = sum(s.role == "library" for s in specs)
        print(
            f"{len(specs)} queries ({len(bases)} base clips x {len(args.distortions)} distortions; "
            f"{known} known, {len(specs) - known} unknown) -> {queries_path}"
        )
        return 0

    if args.command == "index":
        films = require_films(args.workspace, manifest)
        embedder = create_embedder(settings.embedder, settings.device, settings.batch_size)
        library = build_library(manifest, films, embedder, settings, library_dir)
        print(f"library: {library.n_videos} films, {library.n_vectors} vectors -> {library_dir}")
        return 0

    if args.command == "embed":
        films = require_films(args.workspace, manifest)
        specs, _ = load_queries(queries_path)
        embedder = create_embedder(settings.embedder, settings.device, settings.batch_size)
        result = embed_queries(
            specs,
            films,
            embedder,
            settings,
            embeddings_dir,
            queries_digest=_digest(queries_path),
            workers=args.workers,
            shard_size=args.shard_size,
            limit=args.limit,
        )
        store = load_query_store(embeddings_dir)
        print(
            f"embedded {result['embedded']} this run; {len(store.queries)}/{len(specs)} done, "
            f"{len(store.failed)} failed -> {embeddings_dir}"
        )
        return 0

    if args.command == "evaluate":
        specs, queries_meta = load_queries(queries_path)
        store = load_query_store(embeddings_dir)
        if store.meta.get("queries_digest") != _digest(queries_path):
            raise ValueError("the embeddings were made from a different answer key")
        library = Library.load(library_dir)
        runs = [
            RunConfig(
                "v1-shipped",
                V1Voting(min_conf=settings.v1_min_conf, min_vote_ratio=settings.v1_min_vote_ratio),
                tune=False,
            ),
            RunConfig("v1-tuned", V1Voting(), tune=True),
            RunConfig("v2-tuned", OffsetVoting(tolerance_s=settings.v2_tolerance_s), tune=True),
        ]
        summary, results, rows = run_evaluation(
            library,
            specs,
            store,
            runs,
            top_k=settings.top_k,
            target_far=args.target_far,
            bootstrap=args.bootstrap,
            allow_partial=args.allow_partial,
        )
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sceneid": __version__,
            "git": git_revision(),
            "manifest": {
                "name": manifest.name,
                "digest": manifest.digest,
                "n_heldout": len(manifest.heldout),
            },
            "answer_key": {"digest": _digest(queries_path), **queries_meta},
            "embeddings": {k: v for k, v in store.meta.items()},
            "evaluated_on": machine_info(),
            **summary,
        }
        out = args.out or Path("benchmarks/results") / f"{manifest.name}-{settings.embedder}"
        write_results(out, summary, results, rows)
        print(f"results -> {out}")
        for r in results:
            t = {k: 100 * v for k, v in r["test"].items() if isinstance(v, float)}
            print(
                f"  {r['name']:<11} DIR {t.get('dir', 0):5.1f}%  FAR {t.get('far', 0):5.1f}%  "
                f"top-1 {t.get('top1', 0):5.1f}%  "
                f"timestamp<=1s {t.get('offset_within_1s', 0):5.1f}%"
            )
        return 0

    if args.command == "render":
        specs, _ = load_queries(queries_path)
        spec = next((s for s in specs if s.query_id == args.query_id), None)
        if spec is None:
            raise ValueError(f"no query {args.query_id!r}")
        film = load_film_file(args.workspace, spec.film_id)
        if film is None:
            raise ValueError(f"film {spec.film_id} is not downloaded")
        render(
            spec.distortion,
            film.path,
            spec.ref_start_s,
            spec.ref_duration_s,
            args.out,
            seed=spec.seed,
            width=film.width,
            height=film.height,
        )
        print(json.dumps({**spec.__dict__, "file": str(args.out)}, indent=2))
        return 0

    if args.command == "status":
        downloaded = [f.id for f in manifest.films if load_film_file(args.workspace, f.id)]
        print(f"films downloaded: {len(downloaded)}/{len(manifest.films)}")
        if queries_path.is_file():
            specs, _ = load_queries(queries_path)
            print(f"answer key: {len(specs)} queries ({queries_path})")
        else:
            specs = []
            print("answer key: not built")
        if Library.exists(library_dir):
            lib = Library.load(library_dir)
            print(f"library: {lib.n_videos}/{len(manifest.library)} films indexed ({library_dir})")
        else:
            print("library: not built")
        if (embeddings_dir / "meta.json").is_file():
            store = load_query_store(embeddings_dir)
            print(f"embeddings: {len(store.queries)}/{len(specs)} done, {len(store.failed)} failed")
        else:
            print("embeddings: none yet")
        return 0
    return 1
