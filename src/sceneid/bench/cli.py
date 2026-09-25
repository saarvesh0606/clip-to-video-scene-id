"""``sceneid bench``: download films, build the answer key, embed, evaluate.

Typical run (the embed steps want a GPU; see notebooks/benchmark_colab.ipynb):

    sceneid bench download
    sceneid bench queries
    sceneid bench index    --embedders clip-vit-b32 sscd-disc-mixup
    sceneid bench embed    --embedders clip-vit-b32 sscd-disc-mixup
    sceneid bench evaluate --embedders clip-vit-b32 sscd-disc-mixup

Films live in the workspace (fast, disposable disk). Everything worth keeping lives in the
store: the answer key, one library and one query-embedding set per embedder, the rendered
clips, and results.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..config import Settings
from ..embedders import EMBEDDERS, create_embedder
from ..library import Library
from ..matching import OffsetVoting, V1Voting
from .distortions import DISTORTIONS, render
from .download import download_all, load_film_file, require_films
from .embedding import build_libraries, embed_queries, load_query_store, machine_info
from .evaluate import RunConfig, run_evaluation
from .manifest import load_manifest
from .queries import build_queries, load_queries, save_queries
from .report import git_revision, write_results

# V1's shipped thresholds (0.83 / 0.90) were set for CLIP ViT-B/32 only.
SHIPPED_V1_EMBEDDER = "clip-vit-b32"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sceneid bench", description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, default=Path("benchmarks/datasets/tier1.json"))
    p.add_argument("--workspace", type=Path, default=Path("data/bench"), help="where films go")
    p.add_argument("--store", type=Path, help="where results go (default: the workspace)")
    p.add_argument("--queries", type=Path, help="answer key (default: <store>/queries.jsonl)")
    p.add_argument("--clips-dir", type=Path, help="rendered clips (default: <store>/clips)")
    sub = p.add_subparsers(dest="command", required=True)

    def with_embedders(parser):
        parser.add_argument(
            "--embedders", nargs="+", choices=EMBEDDERS, help="default: the --embedder setting"
        )
        return parser

    s = sub.add_parser("download", help="download and verify the films")
    s.add_argument("--only", nargs="+", metavar="FILM_ID")

    s = sub.add_parser("queries", help="build the answer key")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--clips-per-minute", type=float, default=1.0)
    s.add_argument("--distortions", nargs="+", choices=list(DISTORTIONS), default=list(DISTORTIONS))

    with_embedders(sub.add_parser("index", help="index the library films (resumable)"))

    s = with_embedders(sub.add_parser("embed", help="render and embed every query (resumable)"))
    s.add_argument("--workers", type=int, default=2, help="parallel ffmpeg renders")
    s.add_argument("--shard-size", type=int, default=200)
    s.add_argument("--limit", type=int, help="stop after this many queries (for a trial run)")
    s.add_argument(
        "--no-keep-clips", action="store_true", help="delete rendered clips instead of caching"
    )

    s = with_embedders(sub.add_parser("evaluate", help="score each embedder, write reports"))
    s.add_argument("--out-root", type=Path, default=Path("benchmarks/results"))
    s.add_argument("--tag", help="suffix for the results folder name")
    s.add_argument("--target-far", type=float, default=0.01)
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--allow-partial", action="store_true", help="evaluate what is embedded so far")

    s = sub.add_parser("render", help="write one query clip to a file, to look at it")
    s.add_argument("query_id")
    s.add_argument("--out", type=Path, required=True)

    with_embedders(sub.add_parser("status", help="show what is done so far"))
    return p


class _Paths:
    def __init__(self, args, settings: Settings):
        store = args.store or args.workspace
        self.store = store
        self.queries = args.queries or store / "queries.jsonl"
        self.clips = args.clips_dir or store / "clips"
        self.embedders = getattr(args, "embedders", None) or [settings.embedder]

    def library(self, embedder: str) -> Path:
        return self.store / "libraries" / embedder

    def embeddings(self, embedder: str) -> Path:
        return self.store / "embeddings" / embedder


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str], settings: Settings) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    paths = _Paths(args, settings)

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
        save_queries(specs, paths.queries, meta)
        bases = {s.base_id for s in specs}
        known = sum(s.role == "library" for s in specs)
        print(
            f"{len(specs)} queries ({len(bases)} base clips x {len(args.distortions)} distortions; "
            f"{known} known, {len(specs) - known} unknown) -> {paths.queries}"
        )
        return 0

    def load_embedders():
        return [create_embedder(n, settings.device, settings.batch_size) for n in paths.embedders]

    if args.command == "index":
        films = require_films(args.workspace, manifest)
        embedders = load_embedders()
        libraries = build_libraries(
            manifest, films, embedders, settings, {e.name: paths.library(e.name) for e in embedders}
        )
        for name, lib in libraries.items():
            print(f"library {name}: {lib.n_videos} films, {lib.n_vectors} vectors")
        return 0

    if args.command == "embed":
        films = require_films(args.workspace, manifest)
        specs, _ = load_queries(paths.queries)
        embedders = load_embedders()
        result = embed_queries(
            specs,
            films,
            embedders,
            settings,
            {e.name: paths.embeddings(e.name) for e in embedders},
            queries_digest=_digest(paths.queries),
            clips_dir=None if args.no_keep_clips else paths.clips,
            workers=args.workers,
            shard_size=args.shard_size,
            limit=args.limit,
        )
        print(f"processed {result['processed']} queries this run")
        for e in embedders:
            store = load_query_store(paths.embeddings(e.name))
            print(
                f"  {e.name}: {len(store.queries)}/{len(specs)} done, "
                f"{len(store.failed)} failed -> {paths.embeddings(e.name)}"
            )
        return 0

    if args.command == "evaluate":
        specs, queries_meta = load_queries(paths.queries)
        for name in paths.embedders:
            store = load_query_store(paths.embeddings(name))
            if store.meta.get("queries_digest") != _digest(paths.queries):
                raise ValueError(f"{name}: the embeddings were made from a different answer key")
            library = Library.load(paths.library(name))
            runs = [
                RunConfig("v1-tuned", V1Voting(), tune=True),
                RunConfig("v2-tuned", OffsetVoting(tolerance_s=settings.v2_tolerance_s), tune=True),
            ]
            if name == SHIPPED_V1_EMBEDDER:
                shipped = V1Voting(
                    min_conf=settings.v1_min_conf, min_vote_ratio=settings.v1_min_vote_ratio
                )
                runs.insert(0, RunConfig("v1-shipped", shipped, tune=False))
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
                "answer_key": {"digest": _digest(paths.queries), **queries_meta},
                "embeddings": dict(store.meta),
                "evaluated_on": machine_info(),
                **summary,
            }
            folder = f"{manifest.name}-{name}" + (f"-{args.tag}" if args.tag else "")
            out = write_results(args.out_root / folder, summary, results, rows)
            print(f"{name}: results -> {out}")
            for r in results:
                t = {k: 100 * v for k, v in r["test"].items() if isinstance(v, float)}
                print(
                    f"  {r['name']:<11} DIR {t.get('dir', 0):5.1f}%  FAR {t.get('far', 0):5.1f}%  "
                    f"top-1 {t.get('top1', 0):5.1f}%  "
                    f"timestamp<=1s {t.get('offset_within_1s', 0):5.1f}%"
                )
        return 0

    if args.command == "render":
        specs, _ = load_queries(paths.queries)
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
        specs = []
        if paths.queries.is_file():
            specs, _ = load_queries(paths.queries)
            print(f"answer key: {len(specs)} queries ({paths.queries})")
        else:
            print("answer key: not built")
        cached = len(list(paths.clips.glob("*.mp4"))) - len(list(paths.clips.glob("*.part.mp4")))
        print(f"rendered clips cached: {max(cached, 0)} ({paths.clips})")
        for name in paths.embedders:
            if Library.exists(paths.library(name)):
                lib = Library.load(paths.library(name))
                print(f"{name} library: {lib.n_videos}/{len(manifest.library)} films indexed")
            else:
                print(f"{name} library: not built")
            if (paths.embeddings(name) / "meta.json").is_file():
                store = load_query_store(paths.embeddings(name))
                print(
                    f"{name} embeddings: {len(store.queries)}/{len(specs)} done, "
                    f"{len(store.failed)} failed"
                )
            else:
                print(f"{name} embeddings: none yet")
        return 0
    return 1
