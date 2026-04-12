from pathlib import Path
import argparse
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from extract_frames import extract_frames
from PIL import Image
from embed_visual import (
    list_frames,
    parse_timestamp_from_filename,
    save_visual_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--video_id", required=True, help="Unique ID for the video (e.g., test2)")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS to extract frames")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    # output locations (defaults)
    parser.add_argument("--frames_out", default="data/embeddings", help="Base folder for extracted frames")
    parser.add_argument("--emb_out", default="data/embeddings", help="Base folder for npz embeddings")
    parser.add_argument("--faiss_dir", default="data/faiss", help="Folder where visual.index and meta.json live")
    return parser.parse_args()

def ensure_paths(video_id: str, frames_out: str, emb_out: str, faiss_dir: str):
    frames_dir = Path(frames_out) / f"{video_id}_frames"
    emb_path = Path(emb_out) / f"{video_id}_visual.npz"
    index_path = Path(faiss_dir) / "visual.index"
    meta_path = Path(faiss_dir) / "meta.json"
    return frames_dir, emb_path, index_path, meta_path

def load_clip(device: str):
    return SentenceTransformer("clip-ViT-B-32", device=device)

def embed_frames_to_npz(
    frames_dir: Path,
    emb_path: Path,
    model: SentenceTransformer,
    device: str,
):
    frames = list_frames(frames_dir)
    if not frames:
        raise ValueError(f"No frames detected in {frames_dir}")

    # Extract timestamps
    timestamps = np.array(
        [parse_timestamp_from_filename(f.name) for f in frames],
        dtype=np.float32,
    )

    # Load images as PIL (CLIP expects PIL Images)
    images = [Image.open(f).convert("RGB") for f in frames]

    # Encode with CLIP
    embeddings = model.encode(
        images,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    frame_names = np.array([f.name for f in frames])

    # Save
    save_visual_embeddings(
        emb_path,
        embeddings,
        timestamps,
        frame_names,
    )

    print(f"Saved embeddings: {embeddings.shape} -> {emb_path}")

def load_existing_index_and_meta(index_path: Path, meta_path: Path, dim: int = 512):
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index: {index_path} (ntotal={index.ntotal})")
    else:
        index = faiss.IndexFlatIP(dim)
        print("Created new FAISS index")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Loaded metadata: {meta_path} (entries={len(meta)})")
    else:
        meta = {}
        print("🆕 Created new metadata store")

    if meta:
        next_id = max(int(k) for k in meta.keys()) + 1
    else:
        next_id = 0

    return index, meta, next_id

def append_to_index(
    index,
    meta: dict,
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    frame_names: np.ndarray,
    video_id: str,
    start_id: int,
):
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    assert len(embeddings) == len(timestamps) == len(frame_names), \
        "Mismatch between embeddings, timestamps, and frame names"

    index.add(embeddings)

    for i in range(len(embeddings)):
        meta[str(start_id + i)] = {
            "video_id": video_id,
            "timestamp": float(timestamps[i]),
            "frame_name": frame_names[i],
        }

    print(f"Appended {len(embeddings)} vectors to FAISS (IDs {start_id} → {start_id + len(embeddings) - 1})")

    return meta

def save_index_and_meta(index, meta: dict, index_path: Path, meta_path: Path):
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved FAISS index to: {index_path}")
    print(f"✅ Saved metadata to: {meta_path}")


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Resolve output paths
    frames_dir, emb_path, index_path, meta_path = ensure_paths(
        video_id=args.video_id,
        frames_out=args.frames_out,
        emb_out=args.emb_out,
        faiss_dir=args.faiss_dir,
    )

    print(f"\n[1/3] Extracting frames from video")
    extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        fps_extract=args.fps,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        overwrite=args.overwrite,
    )

    print(f"\n[2/3] Embedding frames")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clip(device)

    embed_frames_to_npz(
        frames_dir=frames_dir,
        emb_path=emb_path,
        model=model,
        device=device,
    )
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]
    timestamps = data["timestamps"]
    frame_names = data["frame_names"]

    print(f"   Loaded embeddings: {embeddings.shape}")

    print(f"\n[3/3] Updating FAISS index")

    index, meta, next_id = load_existing_index_and_meta(
        index_path=index_path,
        meta_path=meta_path,
        dim=embeddings.shape[1],
    )

    meta = append_to_index(
        index=index,
        meta=meta,
        embeddings=embeddings,
        timestamps=timestamps,
        frame_names=frame_names,
        video_id=args.video_id,
        start_id=next_id,
    )

    save_index_and_meta(
        index=index,
        meta=meta,
        index_path=index_path,
        meta_path=meta_path,
    )

    print("\n🎉 INDEXING COMPLETE")
    print(f"   Video ID: {args.video_id}")
    print(f"   Frames indexed: {len(embeddings)}")
    print(f"   FAISS total vectors: {index.ntotal}")
    
if __name__ == "__main__":
    main()  
