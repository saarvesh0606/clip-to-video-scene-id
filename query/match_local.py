import os

# ==========================================================
# API MODE SILENCING (Option #2)
# Must be set BEFORE importing sentence_transformers/torch/TF
# ==========================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # 0=all,1=info,2=warn,3=error
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")    # avoid tokenizer thread spam

import json
import re
import argparse
import subprocess
from pathlib import Path
import warnings
import logging
import contextlib
import io

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

    # Output / logging controls
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging (sanity matches, per-frame top1, etc.)",
    )
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Print ONLY the final JSON (recommended for API usage).",
    )

    # ==========================================================
    # NEW: UNKNOWN/REJECTION GATE (Option #2)
    # ==========================================================
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.80,
        help="If final confidence < min_conf => return UNKNOWN (best_video_id=None).",
    )
    parser.add_argument(
        "--min_vote_ratio",
        type=float,
        default=0.35,
        help="If best votes / total votes < min_vote_ratio => return UNKNOWN.",
    )

    return parser.parse_args()


def _safe_print(s: str):
    """
    Windows-safe printing: avoids UnicodeEncodeError cascades.
    """
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("utf-8", "replace").decode("utf-8"))


def ensure_query_frames(video_path: Path, fps: float, reextract: bool = False, debug: bool = False) -> Path:
    """
    Ensure frames exist under: data/query_frames/<query_id>/

    Behavior:
    - If frames exist but are not timestamp-named, rebuild them.
    - If --reextract is passed, rebuild them.
    - Uses indexing/extract_frames.py to guarantee _t..._f... naming.
    - Avoids unnecessary overwrite: only overwrites when reextract=True.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Query video not found: {video_path}")

    query_id = video_path.stem
    frames_dir = Path("data/query_frames") / query_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames_dir.glob("*.jpg"))
    has_timestamp_naming = any(("_t" in p.name and "_f" in p.name) for p in existing)

    if existing and has_timestamp_naming and not reextract:
        if debug:
            _safe_print(f"Frames already exist for {query_id}: {len(existing)} found.")
        return frames_dir

    # If frames exist and we need to rebuild, delete old frames
    if existing and (reextract or not has_timestamp_naming):
        reason = "reextract requested" if reextract else "existing frames not timestamp-named"
        if debug:
            _safe_print(f"[WARN] Rebuilding query frames ({reason}) in: {frames_dir}")
        for p in existing:
            try:
                p.unlink()
            except Exception:
                pass

    if debug:
        _safe_print(f"Extracting query frames -> {frames_dir} (fps={fps})")

    cmd = [
        "python",
        "indexing/extract_frames.py",
        "--video",
        str(video_path),
        "--out",
        str(frames_dir),
        "--fps",
        str(fps),
    ]
    if reextract:
        cmd.append("--overwrite")

    if debug:
        _safe_print("Running: " + " ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if res.returncode != 0:
        if debug:
            _safe_print(res.stdout)
            _safe_print(res.stderr)
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

    if debug:
        _safe_print(f"[OK] Extracted {len(extracted)} frames to {frames_dir}")
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


def empty_result(reason: str):
    """
    Strict schema output even on failure.
    """
    return {
        "best_video_id": None,
        "confidence": 0.0,
        "est_timestamp": None,
        "time_window": [None, None],
        "votes_per_video": {},
        "score_sum_per_video": {},
        "top_evidence": [],
        "reason": reason,
    }


# ==========================================================
# NEW: UNKNOWN/REJECTION GATE (Option #2)
# (ADDED BLOCK ONLY — does not change your existing logic)
# ==========================================================
def apply_unknown_gate(result: dict, min_conf: float, min_vote_ratio: float) -> dict:
    """
    If result looks weak/noisy, return UNKNOWN rather than forcing a label.

    Criteria:
    - confidence < min_conf  OR
    - (best_votes / total_votes) < min_vote_ratio
    """
    best_id = result.get("best_video_id")
    conf = float(result.get("confidence", 0.0) or 0.0)

    votes = result.get("votes_per_video") or {}
    total_votes = sum(int(v) for v in votes.values()) if votes else 0
    best_votes = int(votes.get(best_id, 0)) if best_id else 0
    vote_ratio = (best_votes / total_votes) if total_votes > 0 else 0.0

    if (conf < float(min_conf)) or (vote_ratio < float(min_vote_ratio)):
        return {
            "best_video_id": None,
            "confidence": round(conf, 4),
            "est_timestamp": None,
            "time_window": [None, None],
            "votes_per_video": votes,
            "score_sum_per_video": result.get("score_sum_per_video", {}) or {},
            "top_evidence": result.get("top_evidence", []) or [],
            "reason": f"unknown_below_threshold(conf={conf:.4f}, vote_ratio={vote_ratio:.3f})",
        }

    return result


# -----------------------------
# Aggregation (Votes + Alignment)
# -----------------------------
def aggregate_votes(
    I: np.ndarray,
    D_sim: np.ndarray,
    meta: dict,
    query_times: np.ndarray | None = None,
    window_size: float = 6.0,
    bin_size: float = 0.5,
    inlier_tol: float = 0.75,
):
    """
    Combine results across query frames.

    Output behavior:
    - est_timestamp: GLOBAL aligned timestamp (weighted over all inliers)
    - time_window: best-matching dense segment window (tie-break by score sum)
    """
    n_query, top_k = I.shape

    votes: dict[str, int] = {}
    score_sum: dict[str, float] = {}
    hits_all = []

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
                    "timestamp": ts,
                    "t_query": t_q,
                    "score": score,
                    "faiss_id": idx,
                    "frame_name": fname,
                    "query_frame_i": qi,
                    "rank": r,
                }
            )

    if not votes:
        return empty_result("No valid matches found in metadata.")

    best_video_id = sorted(votes.keys(), key=lambda v: (votes[v], score_sum[v]), reverse=True)[0]

    best_hits = [h for h in hits_all if h["video_id"] == best_video_id]
    best_hits.sort(key=lambda x: x["score"], reverse=True)

    top1_scores = [h["score"] for h in best_hits if h["rank"] == 0]
    if len(top1_scores) == 0:
        base_conf = 0.0
    else:
        top1_scores = np.array(top1_scores, dtype=np.float32)
        mean_score = float(top1_scores.mean())
        strong_ratio = float((top1_scores >= 0.90).mean())
        base_conf = 0.7 * mean_score + 0.3 * strong_ratio

    align_ratio = 0.0
    inliers = best_hits
    best_offset = 0.0

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

        if len(inliers) < max(5, int(0.3 * len(best_hits))):
            inliers = best_hits

        align_ratio = len(inliers) / max(1, len(best_hits))

    confidence = 0.6 * base_conf + 0.4 * align_ratio
    confidence = float(np.clip(confidence, 0.0, 1.0))
    confidence = round(confidence, 4)

    if have_query_times and len(inliers) > 0:
        times_est = np.array([h["t_query"] + best_offset for h in inliers], dtype=np.float32)
    else:
        times_est = np.array([h["timestamp"] for h in inliers], dtype=np.float32)

    scores_est = np.array([h["score"] for h in inliers], dtype=np.float32)

    # est_timestamp: GLOBAL weighted over all inliers
    if len(times_est) > 0:
        w_all = np.maximum(scores_est.astype(np.float64), 1e-6)
        t_all = times_est.astype(np.float64)
        est_ts_global = float((t_all * w_all).sum() / float(w_all.sum()))
        est_ts_global = round(est_ts_global, 2)
    else:
        est_ts_global = None

    # time_window: best segment window, tie-break by score sum
    start_t, end_t = None, None
    if len(times_est) > 0:
        W = float(window_size)
        order = np.argsort(times_est)
        times_sorted = times_est[order]
        scores_sorted = scores_est[order]

        best_i, best_j = 0, 1
        best_count = 1
        best_score_sum = float(scores_sorted[0])

        j = 0
        for i in range(len(times_sorted)):
            while j < len(times_sorted) and times_sorted[j] <= times_sorted[i] + W:
                j += 1

            count = j - i
            score_sum_w = float(scores_sorted[i:j].sum()) if j > i else 0.0

            if (count > best_count) or (count == best_count and score_sum_w > best_score_sum):
                best_i, best_j = i, j
                best_count = count
                best_score_sum = score_sum_w

        cluster_times = times_sorted[best_i:best_j]
        if len(cluster_times) == 0:
            cluster_times = times_sorted

        start_t = round(float(cluster_times[0]), 2)
        end_t = round(float(cluster_times[-1]), 2)

    # Evidence: dedupe
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
        "est_timestamp": est_ts_global,
        "time_window": [start_t, end_t],
        "votes_per_video": votes,
        "score_sum_per_video": {k: round(v, 4) for k, v in score_sum.items()},
        "top_evidence": top_evidence,
        "reason": None,
    }


# ==========================================================
# NEW SECTION ADDED (NO CHANGES TO YOUR EXISTING CODE)
# Core API function that returns dict (can be called by FastAPI)
# ==========================================================
# ==========================================================
# NEW SECTION ADDED (NO CHANGES TO YOUR EXISTING CODE)
# Core API function that returns dict (can be called by FastAPI)
# ==========================================================
def match_video(video_path, params=None) -> dict:
    """
    API-friendly wrapper.

    Usage:
        result = match_video("data/query_clips/test2_query.mp4", {"fps":2, "max_frames":20, ...})
    Returns:
        result_json (dict)
    """
    if params is None:
        params = {}

    # Pull params with sensible defaults (matching your CLI defaults)
    fps = float(params.get("fps", 2.0))
    max_frames = int(params.get("max_frames", 20))
    top_k = int(params.get("top_k", 5))
    reextract = bool(params.get("reextract", False))
    debug = bool(params.get("debug", False))
    json_only = bool(params.get("json_only", True))  # API default
    min_conf = float(params.get("min_conf", 0.80))
    min_vote_ratio = float(params.get("min_vote_ratio", 0.35))

    index_path = Path("data/faiss/visual.index")
    meta_path = Path("data/faiss/meta.json")

    # In API mode, silence warnings/loggers (same behavior as main)
    if json_only and not debug:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.ERROR)
        for name in ["tensorflow", "tf_keras", "transformers", "sentence_transformers"]:
            logging.getLogger(name).setLevel(logging.ERROR)

    silence_ctx = contextlib.redirect_stderr(io.StringIO()) if (json_only and not debug) else contextlib.nullcontext()

    try:
        with silence_ctx:
            video_path = Path(video_path)
            query_frames_dir = ensure_query_frames(video_path, fps, reextract=reextract, debug=debug)

            if not index_path.exists():
                return empty_result(f"FAISS index not found: {index_path}")
            index = faiss.read_index(str(index_path))

            if not meta_path.exists():
                return empty_result(f"Metadata not found: {meta_path}")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_raw = json.load(f)
            meta = {str(k): v for k, v in meta_raw.items()}

            if len(meta) != index.ntotal:
                return empty_result(
                    f"meta.json size ({len(meta)}) != FAISS ntotal ({index.ntotal}). Rebuild or re-sync meta/index."
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer("clip-ViT-B-32", device=device)

            frame_paths = sorted(list(query_frames_dir.glob("*.jpg")), key=lambda p: p.name)
            if not frame_paths:
                return empty_result(f"No .jpg frames found in: {query_frames_dir}")

            selected = frame_paths[: max(1, max_frames)]
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

            D, I = index.search(q_emb, max(1, top_k))
            D = faiss_scores_to_similarity(D, index)

            result = aggregate_votes(I, D, meta, query_times=query_times)
            result = apply_unknown_gate(result, min_conf=min_conf, min_vote_ratio=min_vote_ratio)

            return result

    except Exception as e:
        return empty_result(str(e))



# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # API-ready default: json_only unless debug explicitly requested
    json_only = args.json_only or (not args.debug)

    # In API mode, aggressively silence warnings/loggers
    if json_only and not args.debug:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.ERROR)
        for name in ["tensorflow", "tf_keras", "transformers", "sentence_transformers"]:
            logging.getLogger(name).setLevel(logging.ERROR)

    index_path = Path("data/faiss/visual.index")
    meta_path = Path("data/faiss/meta.json")

    # In API mode, also silence stderr during model load/search to avoid stray prints
    silence_ctx = contextlib.redirect_stderr(io.StringIO()) if (json_only and not args.debug) else contextlib.nullcontext()

    try:
        with silence_ctx:
            query_frames_dir = ensure_query_frames(args.video, args.fps, reextract=args.reextract, debug=args.debug)

            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            index = faiss.read_index(str(index_path))

            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata not found: {meta_path}")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_raw = json.load(f)
            meta = {str(k): v for k, v in meta_raw.items()}

            if len(meta) != index.ntotal:
                raise RuntimeError(
                    f"meta.json size ({len(meta)}) != FAISS ntotal ({index.ntotal}). "
                    "Rebuild or re-sync meta/index."
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer("clip-ViT-B-32", device=device)

            frame_paths = sorted(list(query_frames_dir.glob("*.jpg")), key=lambda p: p.name)
            if not frame_paths:
                raise FileNotFoundError(f"No .jpg frames found in: {query_frames_dir}")

            max_frames = max(1, int(args.max_frames))
            selected = frame_paths[:max_frames]

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

            top_k = max(1, int(args.top_k))
            D, I = index.search(q_emb, top_k)
            D = faiss_scores_to_similarity(D, index)

            result = aggregate_votes(I, D, meta, query_times=query_times)

            # ==========================================================
            # NEW: APPLY UNKNOWN GATE (Option #2)
            # ==========================================================
            result = apply_unknown_gate(
                result,
                min_conf=float(args.min_conf),
                min_vote_ratio=float(args.min_vote_ratio),
            )

        if json_only:
            print(json.dumps(result, ensure_ascii=False))
            return

        # Debug prints (only when --debug used)
        _safe_print(f"FAISS index loaded. Total vectors: {index.ntotal}")
        _safe_print(f"Metadata loaded. Entries: {len(meta)}")
        _safe_print(f"CLIP loaded on {device}")
        _safe_print(f"\nUsing {len(selected)} query frames from: {query_frames_dir}")
        for p in selected[:5]:
            _safe_print(" - " + p.name)
        if len(selected) > 5:
            _safe_print(f" - ... ({len(selected)-5} more)")

        _safe_print("\nFINAL RESULT (Aggregation)")
        _safe_print(json.dumps(result, indent=2, ensure_ascii=False))

        _safe_print("\nTop-1 match per query frame:")
        for qi, p in enumerate(selected):
            idx = int(I[qi, 0])
            score = float(D[qi, 0])
            info = meta.get(str(idx), {})
            _safe_print(
                f"q#{qi:02d} {p.name} -> score={score:.4f} | "
                f"video={info.get('video_id')} | t={info.get('timestamp')} | frame={info.get('frame_name')}"
            )

    except Exception as e:
        result = empty_result(str(e))
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()