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

## Setup
```bash
pip install -r requirements.txt
