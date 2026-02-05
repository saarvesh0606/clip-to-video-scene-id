from operator import index
import os
from pathlib import Path
import json
from unittest import result
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Match query video against indexed videos")
    parser.add_argument("--video", type=Path, required=True, help="Path to query video")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS for query frame extraction")
    return parser.parse_args()


def ensure_query_frames(video_path: Path, fps: float) -> Path:
    query_id = video_path.stem
    frames_dir = Path("data/query_frames") / query_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(frames_dir.glob("*.jpg"))
    if existing_frames:
        print(f"Frames already exist for {query_id}: {len(existing_frames)} found.")
        return frames_dir

    # Extract frames using ffmpeg (requires ffmpeg to be installed)
    print(f"Extracting frames from {video_path} at {fps} fps...")
    cmd = f"ffmpeg -i {video_path} -vf fps={fps} {frames_dir}/frame_%04d.jpg"
    print("Running command:", cmd)
    result = os.system(cmd)
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with code {result}")

    extracted_frames = list(frames_dir.glob("*.jpg"))
    print(f"Extracted {len(extracted_frames)} frames to {frames_dir}")
    return frames_dir


def aggregate_votes(
    I: np.ndarray,
    D: np.ndarray,
    meta: dict,
    window_radius: float = 3.0,
    window_size: float = 6.0,
):

    votes = {}       # video_id -> count
    score_sum = {}   # video_id -> sum(scores)
    hits_best = []   # store all hits for later timestamp estimation

    n_query, top_k = I.shape

    # Collect votes
    for qi in range(n_query):
        for r in range(top_k):
            idx = int(I[qi, r])
            score = float(D[qi, r])

            key = str(idx)
            if key not in meta:
                continue

            vid = meta[key]["video_id"]
            ts = float(meta[key]["timestamp"])
            fname = meta[key].get("frame_name", "")

            votes[vid] = votes.get(vid, 0) + 1
            score_sum[vid] = score_sum.get(vid, 0.0) + score

            hits_best.append({
                "video_id": vid,
                "timestamp": ts,
                "score": score,
                "faiss_id": idx,
                "frame_name": fname,
                "query_frame_i": qi,
                "rank": r,
            })

    if not votes:
        return {
            "best_video_id": None,
            "confidence": 0.0,
            "reason": "No valid matches found in metadata.",
        }

    # Pick best video: first by votes, then by score sum
    best_video_id = sorted(
        votes.keys(),
        key=lambda v: (votes[v], score_sum[v]),
        reverse=True
    )[0]

    # Filter hits for best video only
    best_hits = [h for h in hits_best if h["video_id"] == best_video_id]
    best_hits = sorted(best_hits, key=lambda x: x["score"], reverse=True)

    # Confidence: compare best score_sum vs total
    total_score = sum(score_sum.values())
    conf = (score_sum[best_video_id] / total_score) if total_score > 0 else 0.0

    # Estimate timestamp using weighted average (robust enough)
    weights = np.array([h["score"] for h in best_hits], dtype=np.float32)
    times = np.array([h["timestamp"] for h in best_hits], dtype=np.float32)

    # Avoid divide-by-zero
    times = np.array([h["timestamp"] for h in best_hits], dtype=np.float32)
    times.sort()

    W = window_size
    best_i = 0
    best_j = 0
    j = 0
    for i in range(len(times)):
        while j < len(times) and times[j] <= times[i] + W:
            j += 1
        if (j - i) > (best_j - best_i):
            best_i, best_j = i, j

    cluster = times[best_i:best_j]
    start_t = float(cluster[0])
    end_t = float(cluster[-1])
    est_t = float(np.median(cluster))

    time_window = (max(0.0, start_t), end_t)


    # Keep top evidence (best 10 hits)
    top_evidence = best_hits[:10]

    return {
        "best_video_id": best_video_id,
        "confidence": round(conf, 4),
        "est_timestamp": round(est_t, 2),
        "time_window": (round(time_window[0], 2), round(time_window[1], 2)),
        "votes_per_video": votes,
        "score_sum_per_video": {k: round(v, 4) for k, v in score_sum.items()},
        "top_evidence": top_evidence,
    }

def main():
    # Paths
    args = parse_args()
    index_path = Path("data/faiss/visual.index")
    meta_path = Path("data/faiss/meta.json")
    query_frames_dir = ensure_query_frames(args.video, args.fps)
    

    # Load FAISS index
    index = faiss.read_index(str(index_path))
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Metadata loaded. Entries: {len(meta)}")

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("clip-ViT-B-32", device=device)
    print(f"CLIP loaded on {device}")

    # ---- STEP 2: Embed ONE frame ----
    frame_paths = sorted(list(query_frames_dir.glob("*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {query_frames_dir}")
        print(f"\nUsing {len(selected)} query frames from: {query_frames_dir}")
      

    test_frame = frame_paths[0]
    print(f"\nUsing test frame: {test_frame}")

    img = Image.open(test_frame).convert("RGB")

    emb = model.encode(
        [img],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Embedding shape:", emb.shape)
    print("First 8 values:", emb[0, :8])
    print("L2 norm:", float(np.linalg.norm(emb[0])))

    print("\n✅ STEP 2 SUCCESS: Single-frame embedding computed")

    top_k = 5
    D, I = index.search(np.ascontiguousarray(emb.astype(np.float32)), top_k)

    print("\nTop matches:")
    for rank in range(top_k):
        idx = int(I[0, rank])
        score = float(D[0, rank])
        info = meta.get(str(idx), {})

        video_id = info.get("video_id", "unknown")
        timestamp = info.get("timestamp", None)
        frame_name = info.get("frame_name", "")

        print(f"#{rank+1}: score={score:.4f} | id={idx} | video={video_id} | t={timestamp} | frame={frame_name}")

    print("\n✅ STEP 3 SUCCESS: FAISS search works")
    
    # ---- STEP 4: Embed MULTIPLE query frames ----
    frame_paths = sorted(list(query_frames_dir.glob("*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {query_frames_dir}")
        print(f"\nUsing {len(selected)} query frames from: {query_frames_dir}")


    max_frames = 20
    selected = frame_paths[:max_frames]
    print(f"\nUsing {len(selected)} query frames from: {query_frames_dir}")
    for p in selected[:5]:
        print(" -", p.name)
    if len(selected) > 5:
        print(f" - ... ({len(selected)-5} more)")

    images = [Image.open(p).convert("RGB") for p in selected]

    q_emb = model.encode(
        images,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    q_emb = np.ascontiguousarray(q_emb)

    print("Query embeddings shape:", q_emb.shape)

    # Search top-1 for each query frame (easy inspection)
    top_k = 5
    D, I = index.search(q_emb, top_k)

    result = aggregate_votes(I, D, meta)
    print("\n🎯 FINAL MATCH RESULT")
    print(json.dumps(result, indent=2))

    print("\nTop-1 match per query frame:")
    for qi, p in enumerate(selected):
        idx = int(I[qi, 0])
        score = float(D[qi, 0])
        info = meta.get(str(idx), {})
        print(
            f"q#{qi:02d} {p.name} -> score={score:.4f} | "
            f"video={info.get('video_id')} | t={info.get('timestamp')} | frame={info.get('frame_name')}"
        )

    print("\n✅ STEP 4 SUCCESS: query clip frames are matching into the indexed video")

    result = aggregate_votes(I, D, meta)

    print("\n🎯 FINAL RESULT (Aggregation)")
    print(json.dumps(result, indent=2))

    # ---- SANITY CHECKS ----
    best_vid = result.get("best_video_id")
    if best_vid:
        print("\n🧪 SANITY CHECKS")
        votes = result["votes_per_video"]
        score_sums = result["score_sum_per_video"]

        # sort by votes then score_sum
        ranked = sorted(votes.keys(), key=lambda v: (votes[v], score_sums.get(v, 0.0)), reverse=True)

        print("\nTop candidate videos:")
        for v in ranked[:5]:
            print(f" - {v}: votes={votes[v]}, score_sum={score_sums.get(v, 0.0)}")

        # best evidence stats
        evidence = result.get("top_evidence", [])
        if evidence:
            scores = np.array([e["score"] for e in evidence], dtype=np.float32)
            times = np.array([e["timestamp"] for e in evidence], dtype=np.float32)

            print(f"\nBest video evidence stats (top {len(evidence)} hits):")
            print(f" - score min/mean/max: {scores.min():.4f} / {scores.mean():.4f} / {scores.max():.4f}")
            print(f" - timestamp min/median/max: {times.min():.2f} / {np.median(times):.2f} / {times.max():.2f}")

            # strong-hit ratio
            strong = float((scores >= 0.90).mean()) if len(scores) else 0.0
            print(f" - strong-hit ratio (>=0.90): {strong*100:.1f}%")

if __name__ == "__main__":
    main()
