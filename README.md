# Web-Assisted Clip-to-Video Scene Identification (Hybrid)

This project identifies which **movie or YouTube video** a short clip (3–10s) comes from and localizes the **scene timestamp**.

## Current Status
- [x] Environment + dependencies
- [x] Frame extraction at fixed FPS with timestamps in filenames
- [ ] CLIP embeddings for frames
- [ ] FAISS indexing + retrieval
- [ ] Temporal alignment + confidence scoring
- [ ] Web-assisted candidate generation
- [ ] FastAPI demo API + simple UI

## Data Setup (Required)

This repository does NOT include video data or extracted frames.

### Folder Structure
Create the following folders inside the project root:

```text
data/
├── indexed_videos/     # Movies / YouTube videos to index (mp4)
├── query_clips/        # Short clips to identify (3–10 seconds)
└── embeddings/         # Auto-generated frames & embeddings (created by code)


## Setup
```bash
pip install -r requirements.txt
