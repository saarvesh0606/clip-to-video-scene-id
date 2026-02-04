from pathlib import Path
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from PIL import Image
import torch


def main():
    # Paths
    index_path = Path("data/faiss/visual.index")
    meta_path = Path("data/faiss/meta.json")
    frames_dir = Path("data/embeddings/test1_frames")  # use indexed frames for now

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
    frame_paths = sorted(list(frames_dir.glob("*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

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
    frame_paths = sorted(list(frames_dir.glob("*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    max_frames = 20
    selected = frame_paths[:max_frames]
    print(f"\nUsing {len(selected)} query frames from: {frames_dir}")
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
    D, I = index.search(q_emb, 1)

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

if __name__ == "__main__":
    main()
