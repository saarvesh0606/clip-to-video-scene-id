# Web-Assisted Clip-to-Video Scene Identification (V1)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg)
![Status](https://img.shields.io/badge/Status-Stable%20v1.0-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A **CLIP-based video identification system** that determines **which reference video (movie / YouTube)** a short query clip originates from and estimates the **scene timestamp**, with robust **open-set rejection** (returns `UNKNOWN` when no reliable match exists).

This repository represents a **stable, API-ready V1 baseline**, focused on correctness, reproducibility, and clean system design.

---

## 🔍 What This System Does (V1)

- Indexes **reference videos** offline
- Accepts **short query clips** (≈ 3–10 seconds)
- Identifies:
  - `best_video_id`
  - confidence score
  - estimated scene timestamp
  - matching time window
  - supporting visual evidence
- **Rejects unrelated clips** using confidence + vote-ratio gating

> **Important:**  
> Only indexed videos can be identified.  
> If a query clip does not match any indexed video, the system returns **`UNKNOWN`**.

---

## ✅ Current Status (V1 – Final)

- [x] Environment & dependency setup
- [x] Frame extraction at fixed FPS with timestamps
- [x] CLIP visual embeddings (`clip-ViT-B-32`)
- [x] FAISS indexing and retrieval
- [x] Temporal alignment & confidence scoring
- [x] Open-set rejection (UNKNOWN gate)
- [x] FastAPI inference API (JSON-only output)
- [ ] Web-assisted candidate generation *(V2)*
- [ ] Audio-based matching *(V2)*
- [ ] Large-scale datasets *(V2)*

---

## 📁 Project Structure

    ```
    clip-to-video-id/
    ├── indexing/
    │   ├── extract_frames.py      # Timestamped frame extraction
    │   ├── index_video.py         # Offline indexing of reference videos
    ├── query/
    │   ├── match_local.py         # Matching logic + UNKNOWN gate
    ├── api/
    │   ├── app.py                 # FastAPI service
    ├── data/
    │   ├── indexed_videos/        # Reference videos (source of truth)
    │   ├── query_clips/           # Query clips
    │   ├── query_frames/          # Auto-generated query frames
    │   └── faiss/
    │       ├── visual.index       # FAISS index
    │       └── meta.json          # Vector → frame/video metadata
    ```

    ---

## 📦 Data Setup (Required)

This repository **does NOT include video data**.

Create the following folders:

    ```
    data/
    ├── indexed_videos/     # Reference videos to index (mp4)
    ├── query_clips/        # Short clips to identify
    └── faiss/              # Auto-generated index files
    ```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🧱 Indexing Reference Videos (Offline)

Only reference videos must be indexed.

1. Place videos into:
   ```
   data/indexed_videos/
   ```

2. Run indexing:
   ```bash
   python indexing/index_video.py
   ```

This builds:
- `data/faiss/visual.index`
- `data/faiss/meta.json`

> Indexing is **offline and compute-heavy by design**.

---

## 🔍 Local Query Matching (CLI)

```bash
python query/match_local.py \
  --video data/query_clips/sample.mp4 \
  --json_only
```

### Output (example)

```json
{
  "best_video_id": "movie_01",
  "confidence": 0.97,
  "est_timestamp": 312.4,
  "time_window": [310.0, 315.0],
  "reason": null
}
```

### Rejected (UNKNOWN)

```json
{
  "best_video_id": null,
  "confidence": 0.41,
  "reason": "unknown_below_threshold"
}
```

---

## 🌐 FastAPI Service

### Start API

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

- `POST /match` – upload a video clip
- `POST /match_path` – match using local server path (dev only)
- `/docs` – Swagger UI

---

## 🔒 Frozen Defaults (V1)

| Parameter        | Value |
|------------------|-------|
| fps              | 3     |
| max_frames       | 40    |
| top_k            | 10    |
| min_conf         | 0.83  |
| min_vote_ratio   | 0.90  |

These defaults are **globally enforced** across CLI and API.

---

## 🧠 Core Design Notes

- **Closed-set identification**
  - Can only identify indexed videos
- **Open-set behavior**
  - Weak or unrelated matches return `best_video_id = null`
- **Visual-only (V1)**
  - No audio fingerprints yet
- **No forced matches**
  - FAISS similarity ≠ identity unless confidence is high

---

## ⚠️ Limitations (Explicit)

- Not a YouTube / Content-ID replacement
- Cannot identify arbitrary internet videos
- Dataset size limited by indexed content
- No cross-modal (audio/text) matching in V1

---

## 🚧 Roadmap (V2)

Planned improvements:

- Shot / scene-level indexing (MovieNet-style)
- Audio embeddings
- Robustness benchmarking
- Web-assisted candidate narrowing
- UI layer for demo

---

## 🏷️ Version

**v1.0 – Stable Baseline**

- Matching logic complete
- Open-set rejection enabled
- API-ready and reproducible
