"""Downloading benchmark films and recording exactly what was downloaded.

Each film lands at ``<workspace>/films/<id><ext>`` next to ``<id>.json``, which records the
file's sha256, whether it matched the source's published md5, and its timing metadata. A
film counts as present only once that record exists, so an interrupted download is resumed
or redone, never used half-finished.
"""

import hashlib
import json
import logging
import shutil
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..frames import probe
from .manifest import Film, Manifest

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi", ".mpeg", ".mpg")
_USER_AGENT = "sceneid-benchmark/2.0 (+https://github.com/saarvesh0606/clip-to-video-scene-id)"


class DownloadError(Exception):
    pass


@dataclass(frozen=True)
class FilmFile:
    """A downloaded, verified film."""

    film_id: str
    path: str
    sha256: str
    md5_verified: bool
    duration_s: float
    width: int
    height: int
    # Seconds to add to an ffmpeg `-ss` position to get the same moment in the timestamps
    # our frame sampler reads. Non-zero when the video stream starts after the container.
    pts_shift_s: float


def films_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "films"


def load_film_file(workspace: str | Path, film_id: str) -> FilmFile | None:
    record = films_dir(workspace) / f"{film_id}.json"
    if not record.is_file():
        return None
    info = FilmFile(**json.loads(record.read_text(encoding="utf-8")))
    return info if Path(info.path).is_file() else None


def require_films(workspace: str | Path, manifest: Manifest) -> dict[str, FilmFile]:
    files = {}
    missing = []
    for film in manifest.films:
        info = load_film_file(workspace, film.id)
        if info is None:
            missing.append(film.id)
        else:
            files[film.id] = info
    if missing:
        raise DownloadError(f"films not downloaded yet: {', '.join(missing)}; run `bench download`")
    return files


def download_all(manifest: Manifest, workspace: str | Path, only: list[str] | None = None):
    results = {}
    for film in manifest.films:
        if only and film.id not in only:
            continue
        results[film.id] = download_film(film, workspace)
    return results


def download_film(film: Film, workspace: str | Path, retries: int = 3) -> FilmFile:
    existing = load_film_file(workspace, film.id)
    if existing:
        log.info("download.skip", extra={"film": film.id, "reason": "already verified"})
        return existing
    out_dir = films_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f"{film.id}.download"

    for attempt in range(1, retries + 1):
        try:
            _fetch(film.url, part, film.size_bytes)
            break
        except (OSError, DownloadError) as exc:
            log.warning(
                "download.retry", extra={"film": film.id, "attempt": attempt, "error": str(exc)}
            )
            if attempt == retries:
                raise DownloadError(f"{film.id}: {exc}") from exc
            time.sleep(5 * attempt)

    md5, _ = _hashes(part)
    if film.md5 and md5 != film.md5:
        part.unlink()
        raise DownloadError(f"{film.id}: md5 {md5} does not match the published {film.md5}")

    video = _unpack(film, part, out_dir) if film.archive == "zip" else _rename(film, part, out_dir)
    _, sha256 = _hashes(video)
    info = probe(video)
    record = FilmFile(
        film_id=film.id,
        path=str(video.resolve()),
        sha256=sha256,
        md5_verified=bool(film.md5),
        duration_s=round(info.duration_s, 3),
        width=info.width,
        height=info.height,
        pts_shift_s=_pts_shift(video),
    )
    (out_dir / f"{film.id}.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    log.info(
        "download.done",
        extra={
            "film": film.id,
            "mb": round(video.stat().st_size / 1e6),
            "md5_verified": record.md5_verified,
        },
    )
    return record


def _fetch(url: str, dest: Path, expected_size: int) -> None:
    """Download `url` to `dest`, resuming a partial file with an HTTP Range request."""
    have = dest.stat().st_size if dest.exists() else 0
    if have == expected_size:
        return
    if have > expected_size:
        dest.unlink()
        have = 0
    headers = {"User-Agent": _USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = have and getattr(response, "status", 200) == 206
        mode = "ab" if resumed else "wb"
        written = have if resumed else 0
        next_log = written + expected_size // 10
        with open(dest, mode) as out:
            while block := response.read(1 << 20):
                out.write(block)
                written += len(block)
                if written >= next_log:
                    log.info(
                        "download.progress",
                        extra={"file": dest.name, "pct": round(100 * written / expected_size)},
                    )
                    next_log += expected_size // 10
    size = dest.stat().st_size
    if size != expected_size:
        raise DownloadError(f"got {size} bytes, expected {expected_size}")


def _hashes(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def _rename(film: Film, part: Path, out_dir: Path) -> Path:
    suffix = Path(urllib.request.url2pathname(film.url.rsplit("/", 1)[-1])).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise DownloadError(f"{film.id}: can't tell the video type from {film.url}")
    video = out_dir / f"{film.id}{suffix}"
    part.replace(video)
    return video


def _unpack(film: Film, part: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(part) as z:
        members = [m for m in z.infolist() if Path(m.filename).suffix.lower() in VIDEO_SUFFIXES]
        if not members:
            raise DownloadError(f"{film.id}: the zip holds no video file")
        member = max(members, key=lambda m: m.file_size)
        video = out_dir / f"{film.id}{Path(member.filename).suffix.lower()}"
        with z.open(member) as src, open(video, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    part.unlink()
    return video


def _pts_shift(path: Path) -> float:
    """Container start minus video-stream start, from ffprobe (0 when ffprobe is missing).

    ffmpeg's `-ss` counts from the container start; our frame sampler counts from the video
    stream's first frame. The difference is usually zero and at most a fraction of a second.
    """
    if shutil.which("ffprobe") is None:
        return 0.0
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=start_time:stream=start_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(out)
    fmt = float(data.get("format", {}).get("start_time") or 0.0)
    streams = data.get("streams") or [{}]
    stream = float(streams[0].get("start_time") or 0.0)
    return round(fmt - stream, 4)
