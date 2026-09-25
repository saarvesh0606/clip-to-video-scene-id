"""Command line interface: ``sceneid index | remove | list | match | serve``."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .embedders import EMBEDDERS, create_embedder
from .frames import VideoError
from .indexer import index_video
from .library import Library, LibraryError
from .log import configure_logging
from .matching import ALGORITHMS

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

log = logging.getLogger("sceneid.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sceneid",
        description="Find which indexed video a short clip comes from, and where it starts.",
    )
    p.add_argument("--library", type=Path, help="library directory (default: data/library)")
    p.add_argument("--embedder", choices=EMBEDDERS, help="default: clip-vit-b32")
    p.add_argument("--algorithm", choices=ALGORITHMS, help="decision rule (default: v1)")
    p.add_argument("--log-level", help="DEBUG, INFO, WARNING (default: INFO)")
    p.add_argument("--log-json", action="store_true", help="log JSON lines instead of text")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="add reference videos to the library")
    s.add_argument("videos", nargs="+", type=Path, help="video files, or folders of them")
    s.add_argument("--id", dest="video_id", help="video id (single video only; default: file name)")
    s.add_argument("--fps", type=float, help="frames sampled per second (default: 2)")
    s.add_argument("--replace", action="store_true", help="re-index videos already present")

    s = sub.add_parser("remove", help="remove videos from the library")
    s.add_argument("video_ids", nargs="+")

    s = sub.add_parser("list", help="list indexed videos")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("match", help="identify a clip")
    s.add_argument("clip", type=Path)
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.add_argument("--fps", type=float, help="query frames per second (default: 3)")
    s.add_argument("--top-k", type=int, help="neighbours per query frame (default: 10)")
    s.add_argument("--max-frames", type=int, help="query frames used (default: 40)")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return p


def _settings(args: argparse.Namespace) -> Settings:
    overrides = {
        "library_dir": args.library,
        "embedder": args.embedder,
        "algorithm": args.algorithm,
        "log_level": args.log_level,
        "index_fps": getattr(args, "fps", None) if args.command == "index" else None,
        "query_fps": getattr(args, "fps", None) if args.command == "match" else None,
        "top_k": getattr(args, "top_k", None),
        "query_max_frames": getattr(args, "max_frames", None),
    }
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def _expand(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_SUFFIXES))
        else:
            files.append(p)
    return files


def cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    videos = _expand(args.videos)
    if not videos:
        raise LibraryError("no video files found")
    if args.video_id and len(videos) > 1:
        raise LibraryError("--id only works with a single video")
    embedder = create_embedder(settings.embedder, settings.device, settings.batch_size)
    library = Library.open_or_create(settings.library_dir, embedder.name, embedder.dim)
    failed = 0
    for path in videos:
        try:
            stats = index_video(
                library,
                embedder,
                path,
                video_id=args.video_id,
                fps=settings.index_fps,
                short_side=settings.frame_short_side,
                batch_size=settings.batch_size,
                replace=args.replace,
            )
        except (LibraryError, VideoError) as exc:
            log.error("index.video_failed", extra={"video": str(path), "error": str(exc)})
            failed += 1
            continue
        library.save(settings.library_dir)  # after every video, so a crash loses at most one
        r = stats.record
        print(
            f"indexed {r.video_id}: {r.duration_s:.1f}s, {r.n_vectors} frames, "
            f"{stats.realtime_factor:.1f}x realtime"
        )
    print(f"library: {library.n_videos} videos, {library.n_vectors} vectors")
    return 1 if failed else 0


def cmd_remove(args: argparse.Namespace, settings: Settings) -> int:
    library = Library.load(settings.library_dir)
    for video_id in args.video_ids:
        record = library.remove_video(video_id)
        print(f"removed {record.video_id} ({record.n_vectors} vectors)")
    library.save(settings.library_dir)
    return 0


def cmd_list(args: argparse.Namespace, settings: Settings) -> int:
    if not Library.exists(settings.library_dir):
        videos, embedder = [], None
    else:
        library = Library.load(settings.library_dir)
        videos, embedder = library.videos(), library.embedder
    if args.json:
        print(json.dumps({"embedder": embedder, "videos": [v.__dict__ for v in videos]}, indent=2))
        return 0
    if not videos:
        print(f"no videos indexed in {settings.library_dir}")
        return 0
    print(f"{'video_id':<32} {'duration':>9} {'frames':>7} {'fps':>5}  indexed")
    for v in videos:
        print(
            f"{v.video_id:<32} {v.duration_s:>8.1f}s {v.n_vectors:>7} {v.sample_fps:>5g}  "
            f"{v.indexed_at}"
        )
    print(f"{len(videos)} videos, embedder {embedder}")
    return 0


def cmd_match(args: argparse.Namespace, settings: Settings) -> int:
    from .matcher import Matcher

    result = Matcher.from_settings(settings).identify(args.clip)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    t = result.timings_ms
    if result.status == "match":
        start, end = result.ref_window
        print(f"MATCH  {result.video_id}  clip starts at {result.offset_s:.2f}s")
        print(f"  confidence {result.confidence:.3f}, evidence spans {start:.2f}-{end:.2f}s")
    else:
        best = result.candidate.video_id if result.candidate else "none"
        print(f"UNKNOWN  best candidate {best}, confidence {result.confidence:.3f}")
    print(
        f"  {result.n_query_frames} frames, {t['total']:.0f} ms "
        f"(decode {t['decode']:.0f}, embed {t['embed']:.0f}, search {t['search']:.1f})"
    )
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_config=None, access_log=False
    )
    return 0


COMMANDS = {
    "index": cmd_index,
    "remove": cmd_remove,
    "list": cmd_list,
    "match": cmd_match,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    serving = args.command == "serve"
    configure_logging(
        settings.log_level, json_output=args.log_json or (serving and settings.log_json)
    )
    try:
        return COMMANDS[args.command](args, settings)
    except (LibraryError, VideoError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
