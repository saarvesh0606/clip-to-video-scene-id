from pathlib import Path
import json
import numpy as np
import faiss


def load_embeddings(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    embeddings = data["embeddings"]
    timestamps = data["timestamps"]
    frame_names = data["frame_names"]

    # FAISS expects float32 + contiguous
    embeddings = embeddings.astype(np.float32)
    embeddings = np.ascontiguousarray(embeddings)

    timestamps = timestamps.astype(np.float32)
    frame_names = np.array([str(x) for x in frame_names], dtype=object)

    print(f"✅ Loaded: {npz_path}")
    print(f"   embeddings: {embeddings.shape}, {embeddings.dtype}")
    print(f"   timestamps: {timestamps.shape}, {timestamps.dtype}")
    print(f"   frame_names: {frame_names.shape}")

    return embeddings, timestamps, frame_names


def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"✅ FAISS index built. Total vectors: {index.ntotal}, dim: {d}")
    return index


def build_metadata(frame_names: np.ndarray, timestamps: np.ndarray, video_id: str = "test1"):
    """
    Key by FAISS row id so we can map search results back to frame info.
    """
    meta = {}
    for i, (fname, ts) in enumerate(zip(frame_names, timestamps)):
        meta[str(i)] = {
            "video_id": video_id,
            "timestamp": float(ts),
            "frame_name": str(fname),
        }
    return meta


def save_index_and_meta(index, meta: dict, index_path: Path, meta_path: Path):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved FAISS index to: {index_path}")
    print(f"✅ Saved metadata to: {meta_path}")



def main():
    npz_path = Path("data/embeddings/test1_visual.npz")
    index_path = Path("data/faiss/visual.index")
    meta_path = Path("data/faiss/meta.json")

    embeddings, timestamps, frame_names = load_embeddings(npz_path)
    index = build_faiss_index(embeddings)
    meta = build_metadata(frame_names, timestamps, video_id="test1")
    save_index_and_meta(index, meta, index_path, meta_path)


if __name__ == "__main__":
    main()
