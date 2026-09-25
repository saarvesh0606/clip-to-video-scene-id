"""The list of films a benchmark uses, and where each one comes from."""

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

from ..library import VIDEO_ID_RE

ROLES = ("library", "heldout")


@dataclass(frozen=True)
class Film:
    id: str
    title: str
    year: int
    role: str  # "library": indexed; "heldout": never indexed, its clips must come back unknown
    group: str  # films in the same group look alike; used to break down false accepts
    source: str
    url: str
    size_bytes: int
    md5: str | None
    archive: str | None  # "zip" when the download is a zip holding the video
    license: str
    license_url: str
    notes: str | None = None


@dataclass(frozen=True)
class Manifest:
    name: str
    description: str
    films: tuple[Film, ...]
    digest: str  # sha256 of the manifest file, recorded with every result

    def get(self, film_id: str) -> Film:
        for f in self.films:
            if f.id == film_id:
                return f
        raise KeyError(film_id)

    @property
    def library(self) -> list[Film]:
        return [f for f in self.films if f.role == "library"]

    @property
    def heldout(self) -> list[Film]:
        return [f for f in self.films if f.role == "heldout"]


def load_manifest(path: str | Path) -> Manifest:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    known = {f.name for f in fields(Film)}
    films = []
    for entry in data["films"]:
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"film {entry.get('id')!r} has unknown fields: {sorted(unknown)}")
        films.append(Film(**entry))

    ids = [f.id for f in films]
    if len(set(ids)) != len(ids):
        raise ValueError("film ids must be unique")
    for f in films:
        if not VIDEO_ID_RE.match(f.id):
            raise ValueError(f"film id {f.id!r} is not a valid video id")
        if f.role not in ROLES:
            raise ValueError(f"film {f.id!r}: role must be one of {ROLES}")
        if f.archive not in (None, "zip"):
            raise ValueError(f"film {f.id!r}: archive must be null or 'zip'")
    if not any(f.role == "library" for f in films):
        raise ValueError("a benchmark needs at least one library film")
    return Manifest(
        data["name"], data.get("description", ""), tuple(films), hashlib.sha256(raw).hexdigest()
    )
