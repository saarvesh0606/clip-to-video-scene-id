"""The answer key: which clips to cut, from where, with which distortion, in which split.

Everything is drawn from a seeded generator keyed by film id, so the same seed always gives
the same clips, and adding a film never changes another film's clips.

Each *base clip* (a film, a start time and a length) is rendered once per distortion. All
renderings of a base clip share one split, so a threshold tuned on the validation split has
never seen any version of a test clip. Confidence intervals resample base clips too, since
the ten renderings of one clip are not independent samples.
"""

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .distortions import DISTORTIONS
from .download import FilmFile
from .manifest import Manifest

CLIP_LENGTHS_S = (3.0, 5.0, 10.0)


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    base_id: str
    film_id: str
    role: str  # "library" (should match film_id) or "heldout" (should be unknown)
    group: str
    split: str  # "val" or "test"
    ref_start_s: float  # ffmpeg -ss position in the film
    ref_duration_s: float  # how much of the film the clip covers (length x speed)
    length_s: float  # the clip's own length
    distortion: str
    seed: int
    true_offset_s: float  # where the clip starts, in the frame sampler's timestamps

    @property
    def expected_video_id(self) -> str | None:
        return self.film_id if self.role == "library" else None


def build_queries(
    manifest: Manifest,
    films: dict[str, FilmFile],
    *,
    seed: int = 0,
    distortions: tuple[str, ...] = tuple(DISTORTIONS),
    clips_per_minute: float = 1.0,
    min_clips: int = 4,
    max_clips: int = 60,
    margin_frac: float = 0.08,
) -> list[QuerySpec]:
    """Sample base clips from every film and expand each into one query per distortion.

    The first and last `margin_frac` of each film are skipped: opening titles and end
    credits are near-identical across films and would test nothing but luck.
    """
    unknown = set(distortions) - set(DISTORTIONS)
    if unknown:
        raise ValueError(f"unknown distortions: {sorted(unknown)}")
    max_speed = max(DISTORTIONS[d].speed for d in distortions)
    specs = []
    for film in manifest.films:
        info = films[film.id]
        rng = np.random.default_rng([seed, zlib.crc32(film.id.encode())])
        duration = info.duration_s
        n_clips = int(np.clip(round(duration / 60 * clips_per_minute), min_clips, max_clips))
        lo, hi = duration * margin_frac, duration * (1 - margin_frac)
        # Stratified: split the usable span into equal segments and put one clip at a random
        # spot in each, so clips never overlap and cover the whole film.
        segment = (hi - lo) / n_clips
        placed: list[tuple[float, float]] = []  # (start, length)
        for i in range(n_clips):
            length = float(rng.choice(CLIP_LENGTHS_S))
            room = segment - length * max_speed - 1.0  # leave 1 s between clips
            if room < 0:
                raise ValueError(f"{film.id} is too short for {n_clips} clips")
            placed.append((lo + i * segment + float(rng.uniform(0, room)), length))

        placed.sort()
        splits = np.array(["val", "test"] * (len(placed) // 2 + 1))[: len(placed)]
        rng.shuffle(splits)
        for i, ((start, length), split) in enumerate(zip(placed, splits, strict=True)):
            base_id = f"{film.id}-{i:03d}"
            for d in distortions:
                speed = DISTORTIONS[d].speed
                specs.append(
                    QuerySpec(
                        query_id=f"{base_id}-{d}",
                        base_id=base_id,
                        film_id=film.id,
                        role=film.role,
                        group=film.group,
                        split=str(split),
                        ref_start_s=round(start, 3),
                        ref_duration_s=round(length * speed, 3),
                        length_s=length,
                        distortion=d,
                        seed=int(rng.integers(2**31)),
                        true_offset_s=round(start + info.pts_shift_s, 3),
                    )
                )
    return specs


def save_queries(specs: list[QuerySpec], path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for s in specs:
            f.write(json.dumps(asdict(s)) + "\n")


def load_queries(path: str | Path) -> tuple[list[QuerySpec], dict]:
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        specs = [QuerySpec(**json.loads(line)) for line in f if line.strip()]
    return specs, meta
