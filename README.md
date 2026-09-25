# sceneid: clip-to-video scene identification

![CI](https://github.com/saarvesh0606/clip-to-video-scene-id/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Give it a short clip (3 to 60 seconds). It tells you which video in its library the clip
came from and where in that video the clip starts, or answers `unknown` when the clip isn't
from any indexed video.

```
$ sceneid match clip.mp4
MATCH  test1  clip starts at 11.75s
  confidence 0.950, evidence spans 11.75-17.75s
  18 frames, 6145 ms (decode 1853, embed 4235, search 43.4)
```

## Status

This is the v2 rebuild. v1 worked on a few hand-picked videos but was never measured;
v2 is being built around a reproducible benchmark so every number here comes from a
script in this repo.

| Area | State |
|---|---|
| Library: add, remove, persist, integrity checks | done |
| Frame sampling from presentation timestamps (correct on variable-frame-rate video) | done |
| V1 decision rule, ported exactly and checked against the original code | done |
| HTTP API: upload limits, concurrency cap, request ids, JSON logs, Prometheus metrics | done |
| CLI, test suite (no model download needed), CI | done |
| Benchmark: 18 open films, ~6,200 distorted clips, open-set metrics with confidence intervals | built; first run pending |
| V2 algorithm: per-video temporal alignment, ratio test, sub-second offsets | built; thresholds provisional |
| V2 thresholds chosen from the benchmark's ROC curve, then V2 becomes the default | after the benchmark |
| Embedder comparison: CLIP, DINOv2, SSCD, perceptual-hash baseline | planned |
| Hosted demo | planned |

**No benchmark results yet.** The benchmark is built and tested; its first full run is
next, and its report will be published in [`benchmarks/results/`](benchmarks/results).

## How it works

```mermaid
flowchart LR
    subgraph offline["Indexing (offline)"]
        V[reference video] --> S1[sample 2 fps] --> E1[embed frames] --> L[(library<br/>FAISS + metadata)]
    end
    subgraph online["Matching (per clip)"]
        C[query clip] --> S2[sample 3 fps] --> E2[embed frames] --> K[top-k search]
        L --> K --> D[decide: video, offset,<br/>or unknown]
    end
```

1. **Sample.** Frames are taken on a fixed grid (every 0.5 s when indexing) using each
   frame's presentation timestamp, so timestamps stay correct on phone recordings.
2. **Embed.** Each frame becomes a unit vector (CLIP ViT-B/32 by default), so inner
   product equals cosine similarity.
3. **Search.** Exact inner-product search in FAISS finds the nearest reference frames for
   every query frame.
4. **Decide.** The algorithm turns those neighbours into an answer: the video, the offset
   where the clip starts (`reference time - query time`), and a confidence. Below the
   confidence gate the answer is `unknown`.

The library records which embedder built it and refuses to be queried with another, and
it cross-checks its files on load, so a stale or half-written library fails loudly
instead of returning wrong answers.

### Decision rules: v1 (baseline) and v2

The v1 rule is kept, unchanged, as the baseline ([`matching/v1.py`](src/sceneid/matching/v1.py)),
so the benchmark can measure its flaws instead of guessing:

- It picks the video by raw vote count over all top-k neighbours, before checking that
  the matches line up in time.
- When too few matches agree on one offset it falls back to using all of them, which
  sets the alignment score to its maximum. On the sample videos this fallback fired for
  both a correct clip and an unrelated one.
- Its vote-ratio gate counts every neighbour, so it gets harder to pass as the library
  grows, right answer or not.
- Its timestamp is the centre of a half-second histogram bin, so it is up to 0.25 s off
  even when everything else is right.

v2 ([`matching/v2.py`](src/sceneid/matching/v2.py)) fixes each of these. For every video it
finds the offset at which the most query frames line up, counting each frame once; frames
that don't line up count for nothing. A video's score is the clip's average similarity at
that offset. The best video must beat the runner-up by a margin (a ratio test, independent
of library size), and the offset is the weighted median of the aligned frames, not a bin.

v2's thresholds are placeholders until the benchmark picks them from a validation ROC
curve, so v1 stays the default for now (`--algorithm v2` or `SCENEID_ALGORITHM=v2` to try it).

## Benchmark

**Films.** 18 films that are free to cut up and show ([manifest](benchmarks/datasets/tier1.json),
with licences and checksums). The system indexes 13 of them (≈9.4 h). The other 5 (≈5.1 h) are
never indexed: their clips must come back `unknown`, and each one is a look-alike of an indexed
film (another Caminandes episode, another 1945 film noir, another early Technicolor film, ...).

| Source | Films | Licence |
|---|---|---|
| Blender Foundation open movies | Big Buck Bunny, Sintel, Tears of Steel, Elephants Dream, Cosmos Laundromat, Caminandes 1-3, Sprite Fright | CC BY |
| Internet Archive feature films | Night of the Living Dead, His Girl Friday, The Little Shop of Horrors, Detour, The General, Royal Wedding, Carnival of Souls, Scarlet Street, A Star Is Born | US public domain |

**Clips.** A seeded script places about one 3, 5 or 10 s clip per minute of film (skipping opening
titles and end credits) and renders each one ten ways: `original`, `compression`, `downscale` (240p),
`crop`, `letterbox`, `mirror`, `color`, `overlay` (logo and caption bar), `speed` (1.25x) and
`screen_recording` (bezel, tilt, blur, noise). That's about 620 base clips and 6,200 queries.

**Scoring.** Open-set identification metrics: correctly identified (DIR), false accepts on unknown
clips (FAR), rejections, wrong-film accepts, top-1 before the gate, and timestamp error. Thresholds
are chosen on the val split (highest DIR with FAR ≤ 1%) and every reported number is from the
separate test split. 95% intervals come from resampling base clips, since the ten versions of one
clip aren't independent. Results are broken down by distortion, clip length, look-alike group and
film, with per-stage latency and index size.

**Running it.** Embedding needs a GPU, so the full run is a
[Colab notebook](notebooks/benchmark_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/saarvesh0606/clip-to-video-scene-id/blob/main/notebooks/benchmark_colab.ipynb)).
Every step is resumable. The same steps from a shell:

```bash
pip install -e ".[clip,bench]"   # plus ffmpeg on PATH
sceneid bench download           # ~7 GB, checksums verified
sceneid bench queries            # the answer key
sceneid bench index              # the library films
sceneid bench embed              # render + embed every clip (GPU recommended)
sceneid bench evaluate           # report -> benchmarks/results/tier1-clip-vit-b32/
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[clip,api]"

sceneid index path/to/videos/   # every video in the folder
sceneid list
sceneid match clip.mp4          # add --json for the full result
sceneid serve                   # API on http://127.0.0.1:8000, docs at /docs
```

All settings live in [`config.py`](src/sceneid/config.py) and can be overridden with
`SCENEID_*` environment variables, e.g. `SCENEID_QUERY_FPS=2`.

## HTTP API

| Method | Path | |
|---|---|---|
| `POST` | `/v1/match` | multipart upload `clip`; returns the match result |
| `GET` | `/v1/videos` | indexed videos |
| `GET` | `/health` | liveness |
| `GET` | `/health/ready` | readiness: model and library loaded, library size |
| `GET` | `/metrics` | Prometheus metrics |

`unknown` is a normal `200` answer. Errors use one shape and carry the request id:

```json
{"error": {"code": "clip_too_long", "message": "clip is 94.0s; the limit is 60s"},
 "request_id": "3f2a9c1d0b7e4a55"}
```

| Status | Code | When |
|---|---|---|
| 413 | `payload_too_large` | upload over `SCENEID_MAX_UPLOAD_MB` (default 50) |
| 415 | `unsupported_media_type` | not `.mp4 .mov .m4v .mkv .webm .avi` |
| 422 | `unreadable_video`, `clip_too_long` | can't decode it, or over `SCENEID_MAX_CLIP_SECONDS` |
| 503 | `busy` | every matching slot stayed busy for `SCENEID_QUEUE_TIMEOUT_S` |

Matching is CPU-bound, so it runs in worker threads behind a semaphore
(`SCENEID_MAX_CONCURRENT_MATCHES`, default 2) instead of blocking the event loop.

### Observability

- **Logs** are JSON lines with a request id (taken from `X-Request-ID` or generated, and
  echoed back). Every match logs its outcome, confidence, offset and per-stage timings.
- **Metrics** at `/metrics`: request count and latency per route, match outcomes
  (`match`, `unknown`, `bad_input`, `busy`, `error`), time per stage (decode, embed,
  search, decide), confidence distribution, upload sizes, in-flight matches, library size.

## Development

```bash
pip install -e ".[dev]"
pytest              # uses a tiny built-in embedder and synthetic videos: no model download
ruff check . && ruff format --check .
```

`tests/test_v1_parity.py` runs the ported v1 rule and the original v1 code
([`tests/legacy`](tests/legacy)) on 400 random inputs and requires identical answers.

## Layout

```
src/sceneid/
  frames.py        video probing and timestamped frame sampling
  embedders/       CLIP, and a tiny thumbnail embedder (tests and a no-ML baseline)
  library.py       FAISS index + per-vector metadata, persistence, integrity checks
  indexer.py       adding videos to a library
  matcher.py       sample -> embed -> search -> decide, with per-stage timings
  matching/        decision rules: v1 (baseline, unchanged) and v2
  api/             FastAPI app and Prometheus metrics
  bench/           the benchmark: downloads, answer key, distortions, embedding, scoring, report
  cli.py           the `sceneid` command
benchmarks/        dataset manifests and published results
notebooks/         the Colab notebook that runs the benchmark
tests/             unit, API, CLI and parity tests
```

## License

MIT
