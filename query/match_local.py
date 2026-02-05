import os
import json
import re
import argparse
import subprocess
from pathlib import Path

import numpy as np
import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


# -----------------------------
# Helpers
# -----------------------------
def parse_t_from_name(name: str) -> float:
    """
    Extract timestamp (seconds) from filename like:
    test1_query_t0000003.99_f000000100.jpg

    Supports both integer and decimal timestamps.
    """
    m = re.search(r"_t(\d+(?:\.\d+)?)_f", name)
    if not m:
        raise ValueError(f"Bad timestamp filename (expected _t..._f...): {name}")
    return float(m.group(1))


def parse_args():
    parser = argparse.ArgumentParser(description="Match query video against indexed videos")
    parser.add_argument("--video", type=Path, required=True, help="Path to query video")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS for query frame extraction")
    parser.add_argument("--max_frames", type=int, default=20, help="How many query frames to use")
    parser.add_argument("--top_k", type=int, default=5, help="FAISS top-k per query frame")
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Force re-extract query frames even if they already exist",
    )
    return parser.parse_args()


def ensure_query_frames(video_path: Path, fps: float, reextract: bool = False) -> Path:
    """
    Ensure frames exist under: data/query_frames/<query_id>/

    Behavior:
    - If frames exist but are not timestamp-named, rebuild them.
    - If --reextract is passed, rebuild them.
    - Uses indexing/extract_frames.py to guarantee _t..._f... naming.
    - Subprocess output printing is UTF-8 safe on Windows.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Query video not found: {video_path}")

    query_id = video_path.stem
    frames_dir = Path("data/query_frames") / query_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames_dir.glob("*.jpg"))

    has_timestamp_naming = any(("_t" in p.name and "_f" in p.name) for p in existing)

    if existing and has_timestamp_naming and not reextract:
        print(f"Frames already exist for {query_id}: {len(existing)} found.")
        return frames_dir

    if existing and (reextract or not has_timestamp_naming):
        reason = "reextract requested" if reextract else "existing frames not timestamp-named"
        print(f"[WARN] Rebuilding query frames ({reason}) in: {frames_dir}")
        for p in existing:
            try:
                p.unlink()
            except Exception:
                pass

    print(f"Extracting query frames -> {frames_dir} (fps={fps})")
    cmd = [
        "python",
        "indexing/extract_frames.py",
        "--video",
        str(video_path),
        "--out",
        str(frames_dir),
        "--fps",
        str(fps),
        "--overwrite",
    ]
    if reextract:
        cmd.append("--overwrite")
    print("Running:", " ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if res.returncode != 0:
        try:
            print(res.stdout)
            print(res.stderr)
        except UnicodeEncodeError:
            print(res.stdout.encode("utf-8", "replace").decode("utf-8"))
            print(res.stderr.encode("utf-8", "replace").decode("utf-8"))
        raise RuntimeError(f"Frame extraction failed (code={res.returncode}).")

    extracted = sorted(frames_dir.glob("*.jpg"))
    if not extracted:
        raise RuntimeError(f"No frames extracted into: {frames_dir}")

    bad = [p.name for p in extracted if ("_t" not in p.name or "_f" not in p.name)]
    if bad:
        raise RuntimeError(
            "Extracted frames are not timestamp-named. "
            "Ensure extract_frames.py uses _t..._f... filenames. Example bad: "
            + bad[0]
        )

    print(f"[OK] Extracted {len(extracted)} frames to {frames_dir}")
    return frames_dir


def faiss_scores_to_similarity(D: np.ndarray, index) -> np.ndarray:
    """
    Convert FAISS returned distances/scores into similarity in [0,1] when possible.

    - If index metric is L2: D is (squared) L2 distance for normalized vectors in [0..4].
      Map to sim ~= 1 - D/4, clipped.
    - If index metric is IP/cosine: D is already similarity-like (higher better).
    """
    try:
        metric = index.metric_type
    except Exception:
        return D

    if metric == faiss.METRIC_L2:
        sim = 1.0 - (D / 4.0)
        return np.clip(sim, 0.0, 1.0)
    return D


# -----------------------------
# Aggregation (Votes + Alignment)
# -----------------------------
def aggregate_votes(
    I: np.ndarray,
    D_sim: np.ndarray,
    meta: dict,
    query_times: np.ndarray | None = None,
    window_size: float = 6.0,      # seconds (used for densest-time cluster)
    bin_size: float = 0.5,         # seconds (offset histogram bin)
    inlier_tol: float = 0.75,      # seconds (offset inlier threshold)
):
    """
    Combine results across query frames.

    Output behavior (as requested):
    - est_timestamp: GLOBAL aligned timestamp (weighted over all inliers)
    - time_window: best-matching dense segment window (tie-break by score sum)
    """
    n_query, top_k = I.shape

    votes: dict[str, int] = {}
    score_sum: dict[str, float] = {}
    hits_all = []

    # Collect hits
    for qi in range(n_query):
        t_q = float(query_times[qi]) if query_times is not None else None

        for r in range(top_k):
            idx = int(I[qi, r])
            score = float(D_sim[qi, r])

            info = meta.get(str(idx))
            if not info:
                continue

            vid = info.get("video_id", "unknown")
            ts = float(info.get("timestamp", 0.0))
            fname = info.get("frame_name", "")

            votes[vid] = votes.get(vid, 0) + 1
            score_sum[vid] = score_sum.get(vid, 0.0) + score

            hits_all.append(
                {
                    "video_id": vid,
                    "timestamp": ts,      # db timestamp
                    "t_query": t_q,       # query timestamp (if available)
                    "score": score,       # similarity
                    "faiss_id": idx,
                    "frame_name": fname,
                    "query_frame_i": qi,
                    "rank": r,
                }
            )

    if not votes:
        return {
            "best_video_id": None,
            "confidence": 0.0,
            "reason": "No valid matches found in metadata.",
        }

    # Pick best video: votes first, then score_sum
    best_video_id = sorted(votes.keys(), key=lambda v: (votes[v], score_sum[v]), reverse=True)[0]

    # Hits for best video only
    best_hits = [h for h in hits_all if h["video_id"] == best_video_id]
    best_hits.sort(key=lambda x: x["score"], reverse=True)

    # ---- Base confidence (quality of top-1 matches) ----
    top1_scores = [h["score"] for h in best_hits if h["rank"] == 0]
    if len(top1_scores) == 0:
        base_conf = 0.0
    else:
        top1_scores = np.array(top1_scores, dtype=np.float32)
        mean_score = float(top1_scores.mean())
        strong_ratio = float((top1_scores >= 0.90).mean())
        base_conf = 0.7 * mean_score + 0.3 * strong_ratio  # 0..1

    # ---- Temporal alignment (offset clustering) ----
    align_ratio = 0.0
    inliers = best_hits
    best_offset = 0.0  # IMPORTANT: always defined

    have_query_times = (
        query_times is not None
        and len(best_hits) > 0
        and all(h["t_query"] is not None for h in best_hits)
    )

    if have_query_times:
        offsets = np.array([h["timestamp"] - h["t_query"] for h in best_hits], dtype=np.float32)

        bins = np.floor(offsets / bin_size).astype(int)
        bin_counts = {}
        for b in bins:
            bin_counts[b] = bin_counts.get(b, 0) + 1

        best_bin = max(bin_counts, key=bin_counts.get)
        best_offset = float((best_bin + 0.5) * bin_size)

        inliers = []
        for h in best_hits:
            off = h["timestamp"] - h["t_query"]
            if abs(off - best_offset) <= inlier_tol:
                inliers.append(h)

        # Fallback if alignment is weak
        if len(inliers) < max(5, int(0.3 * len(best_hits))):
            inliers = best_hits

        align_ratio = len(inliers) / max(1, len(best_hits))

    # Combine confidence
    confidence = 0.6 * base_conf + 0.4 * align_ratio
    confidence = float(np.clip(confidence, 0.0, 1.0))
    confidence = round(confidence, 4)

    # ---- Build aligned times for inliers ----
    if have_query_times and len(inliers) > 0:
        times_est = np.array([h["t_query"] + best_offset for h in inliers], dtype=np.float32)
    else:
        times_est = np.array([h["timestamp"] for h in inliers], dtype=np.float32)

    scores_est = np.array([h["score"] for h in inliers], dtype=np.float32)

    # ---- est_timestamp (GLOBAL): weighted over ALL inliers ----
    if len(times_est) > 0:
        w_all = np.maximum(scores_est.astype(np.float64), 1e-6)
        t_all = times_est.astype(np.float64)
        est_ts_global = float((t_all * w_all).sum() / float(w_all.sum()))
    else:
        est_ts_global = 0.0

    # ---- time_window (BEST SEGMENT): densest window, tie-break by score sum ----
    W = float(window_size)

    order = np.argsort(times_est)
    times_sorted = times_est[order]
    scores_sorted = scores_est[order]

    best_i, best_j = 0, 1
    best_count = 1 if len(times_sorted) > 0 else 0
    best_score_sum = float(scores_sorted[0]) if len(scores_sorted) > 0 else 0.0

    j = 0
    for i in range(len(times_sorted)):
        while j < len(times_sorted) and times_sorted[j] <= times_sorted[i] + W:
            j += 1

        count = j - i
        score_sum_w = float(scores_sorted[i:j].sum()) if j > i else 0.0

        # Primary: max count, Secondary: max score_sum
        if (count > best_count) or (count == best_count and score_sum_w > best_score_sum):
            best_i, best_j = i, j
            best_count = count
            best_score_sum = score_sum_w

    cluster_times = times_sorted[best_i:best_j]
    if len(cluster_times) == 0:
        cluster_times = times_sorted

    start_t = float(cluster_times[0]) if len(cluster_times) else 0.0
    end_t = float(cluster_times[-1]) if len(cluster_times) else 0.0

    # Evidence: dedupe by (faiss_id, timestamp)
    seen = set()
    top_evidence = []
    for h in best_hits:
        key = (int(h["faiss_id"]), round(float(h["timestamp"]), 2))
        if key in seen:
            continue
        seen.add(key)
        top_evidence.append(h)
        if len(top_evidence) == 10:
            break

    return {
        "best_video_id": best_video_id,
        "confidence": confidence,
        "est_timestamp": round(est_ts_global, 2),
        "time_window": [round(max(0.0, start_t), 2), round(end_t, 2)],
        "votes_per_video": votes,
        "score_sum_per_video": {k: round(v, 4) for k, v in score_sum.items()},
        "top_evidence": top_evidence,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    index_path = Path("data/faiss/visual.index")
    meta_path = Path("data/faiss/meta.json")

    query_frames_dir = ensure_query_frames(args.video, args.fps, reextract=args.reextract)

    # Load FAISS index
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")

    # Load metadata
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    meta = {str(k): v for k, v in meta_raw.items()}
    print(f"Metadata loaded. Entries: {len(meta)}")

    if len(meta) != index.ntotal:
        raise RuntimeError(
            f"meta.json size ({len(meta)}) != FAISS ntotal ({index.ntotal}). "
            "Rebuild or re-sync meta/index."
        )

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("clip-ViT-B-32", device=device)
    print(f"CLIP loaded on {device}")

    # List query frames
    frame_paths = sorted(list(query_frames_dir.glob("*.jpg")), key=lambda p: p.name)
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {query_frames_dir}")

    # ---- STEP 2: Embed ONE frame ----
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
    emb = np.ascontiguousarray(emb)

    print("Embedding shape:", emb.shape)
    print("First 8 values:", emb[0, :8])
    print("L2 norm:", float(np.linalg.norm(emb[0])))
    print("\nSTEP 2 SUCCESS: Single-frame embedding computed")

    # ---- STEP 3: Single-frame FAISS sanity ----
    top_k_single = min(int(args.top_k), 10)
    D1, I1 = index.search(emb, top_k_single)
    D1 = faiss_scores_to_similarity(D1, index)

    print("\nTop matches:")
    for rank in range(top_k_single):
        idx = int(I1[0, rank])
        score = float(D1[0, rank])
        info = meta.get(str(idx), {})
        print(
            f"#{rank+1}: score={score:.4f} | id={idx} | "
            f"video={info.get('video_id')} | t={info.get('timestamp')} | frame={info.get('frame_name')}"
        )

    print("\nSTEP 3 SUCCESS: FAISS search works")

    # ---- STEP 4: Multi-frame matching + aggregation ----
    max_frames = max(1, int(args.max_frames))
    selected = frame_paths[:max_frames]
    print(f"\nUsing {len(selected)} query frames from: {query_frames_dir}")
    for p in selected[:5]:
        print(" -", p.name)
    if len(selected) > 5:
        print(f" - ... ({len(selected)-5} more)")

    query_times = np.array([parse_t_from_name(p.name) for p in selected], dtype=np.float32)

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

    top_k = max(1, int(args.top_k))
    D, I = index.search(q_emb, top_k)
    D = faiss_scores_to_similarity(D, index)

    result = aggregate_votes(I, D, meta, query_times=query_times)

    print("\nFINAL RESULT (Aggregation)")
    print(json.dumps(result, indent=2))

    # Optional debug print: top-1 per query frame
    print("\nTop-1 match per query frame:")
    for qi, p in enumerate(selected):
        idx = int(I[qi, 0])
        score = float(D[qi, 0])
        info = meta.get(str(idx), {})
        print(
            f"q#{qi:02d} {p.name} -> score={score:.4f} | "
            f"video={info.get('video_id')} | t={info.get('timestamp')} | frame={info.get('frame_name')}"
        )

    print("\nSTEP 4 SUCCESS: query clip frames are matching into the indexed video")


if __name__ == "__main__":
    main()
