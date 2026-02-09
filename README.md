# Web-Assisted Clip-to-Video Scene Identification (V1)

A CLIP-based system that identifies **which reference video (movie / YouTube)** a short query clip comes from and estimates the **scene timestamp**, with **open-set rejection** (returns UNKNOWN if no match exists).

This is a **closed-set, API-ready V1 baseline** focused on correctness, robustness, and clarity.

--------------------------------------------------------------------

WHAT THIS SYSTEM DOES (V1)

- Indexes reference videos offline
- Accepts a short query clip (≈3–10 seconds)
- Identifies:
  - best_video_id
  - confidence
  - estimated timestamp
  - time window
  - supporting evidence
- Rejects unrelated clips using confidence + vote-ratio gating

NOTE:
Only indexed videos can be identified.
If a query does not match any indexed video, the system returns UNKNOWN.

--------------------------------------------------------------------

CURRENT STATUS (V1 – FINAL)

[x] Environment & dependencies
[x] Frame extraction at fixed FPS with timestamps
[x] CLIP visual embeddings (clip-ViT-B-32)
[x] FAISS indexing & retrieval
[x] Temporal alignment & confidence scoring
[x] Open-set rejection (UNKNOWN gate)
[x] FastAPI demo API (JSON-only output)
[ ] Web-assisted candidate generation (V2)
[ ] Audio matching (V2)
[ ] Large-scale datasets (V2)

--------------------------------------------------------------------

PROJECT STRUCTURE

clip-to-video-id/
│
├── indexing/
│   ├── extract_frames.py      # Timestamped frame extraction
│   ├── index_video.py         # Offline indexing of reference videos
│
├── query/
│   ├── match_local.py         # Matching logic + UNKNOWN gate
│
├── api/
│   ├── app.py                 # FastAPI service
│
├── data/
│   ├── indexed_videos/        # Reference videos (source of truth)
│   ├── query_clips/           # Query clips
│   ├── query_frames/          # Auto-generated query frames
│   └── faiss/
│       ├── visual.index       # FAISS index
│       └── meta.json          # Vector → frame/video metadata

--------------------------------------------------------------------

DATA SETUP (REQUIRED)

This repository does NOT include video data.

Create the following folders:

data/
├── indexed_videos/     # Reference videos to index (mp4)
├── query_clips/        # Short clips to identify
└── faiss/              # Auto-generated index files

--------------------------------------------------------------------

SETUP

pip install -r requirements.txt

--------------------------------------------------------------------

INDEXING REFERENCE VIDEOS (OFFLINE)

Only reference videos must be indexed.

1) Place videos into:
   data/indexed_videos/

2) Run indexing:
   python indexing/index_video.py

This builds:
- data/faiss/visual.index
- data/faiss/meta.json

Indexing is OFFLINE and compute-heavy by design.

--------------------------------------------------------------------

LOCAL QUERY MATCHING (CLI)

python query/match_local.py \
  --video data/query_clips/sample.mp4 \
  --json_only

OUTPUT (EXAMPLE)

{
  "best_video_id": "movie_01",
  "confidence": 0.97,
  "est_timestamp": 312.4,
  "time_window": [310.0, 315.0],
  "reason": null
}

REJECTED (UNKNOWN)

{
  "best_video_id": null,
  "confidence": 0.41,
  "reason": "unknown_below_threshold"
}

--------------------------------------------------------------------

FASTAPI SERVICE

START API

python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

ENDPOINTS

POST /match       -> upload a video clip
POST /match_path  -> match using local server path (dev only)
/docs             -> Swagger UI

--------------------------------------------------------------------

FROZEN DEFAULTS (V1)

fps            = 3
max_frames     = 40
top_k          = 10
min_conf       = 0.83
min_vote_ratio = 0.90

These defaults are globally enforced across CLI and API.

--------------------------------------------------------------------

CORE DESIGN NOTES

- Closed-set identification
  - Can only identify indexed videos
- Open-set behavior
  - Weak or unrelated matches return best_video_id = null
- Visual-only (V1)
  - No audio fingerprints yet
- No forced matches
  - FAISS similarity ≠ identity unless confidence is high

--------------------------------------------------------------------

LIMITATIONS (EXPLICIT)

- Not a YouTube / Content-ID replacement
- Cannot identify arbitrary internet videos
- Dataset size limited by indexed content
- No cross-modal (audio/text) matching in V1

--------------------------------------------------------------------

ROADMAP (V2)

Planned improvements:
- Shot / scene-level indexing (MovieNet-style)
- Audio embeddings
- Robustness benchmarking
- Web-assisted candidate narrowing
- UI layer for demo

--------------------------------------------------------------------

VERSION

v1.0 – Stable Baseline

- Matching logic complete
- Open-set rejection enabled
- API-ready and reproducible
