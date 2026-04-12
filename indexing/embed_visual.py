from pathlib import Path
import re
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch


def parse_timestamp_from_filename(name: str) -> float:
    match = re.search(r"_t([0-9]+\.[0-9]+)_f", name)
    if not match:
        raise ValueError(f"Filename '{name}' does not contain a valid timestamp pattern.")
    return float(match.group(1))


def list_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frames = list(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    frames = sorted(frames, key=lambda p: parse_timestamp_from_filename(p.name))
    return frames


def load_clip_model(device: str = "cpu") -> SentenceTransformer:
    model_name = "clip-ViT-B-32"
    model = SentenceTransformer(model_name, device=device)
    print(f"✅ Loaded CLIP model '{model_name}' on device '{device}'")
    return model


def embed_frames(
    frame_paths: list[Path],
    model: SentenceTransformer,
    device: str = "cpu",
    max_frames: int = 5,
) -> np.ndarray:
    """
    Compute real CLIP embeddings for a small set of frames.
    """
    selected = frame_paths[:max_frames]

    print("\nSelected frames:")
    for p in selected:
        print(" -", p.name)

    # Load images as PIL RGB
    images = []
    for p in selected:
        img = Image.open(p).convert("RGB")
        images.append(img)

    # Encode images -> embeddings
    embeddings = model.encode(
        images,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embeddings


def save_visual_embeddings(out_path: Path, embeddings: np.ndarray, timestamps: np.ndarray, frame_names: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        timestamps=timestamps.astype(np.float32),
        frame_names=frame_names,
    )
    print(f"✅ Saved embeddings to: {out_path}")


def main():
    frames_dir = Path("data/embeddings/test1_frames")

    frames = list_frames(frames_dir)
    print(f"Found {len(frames)} frames in {frames_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clip_model(device=device)

    embeddings = embed_frames(
    frame_paths=frames,
    model=model,
    device=device,
    max_frames=len(frames), 
    )

    print("Embeddings shape:", embeddings.shape)
    timestamps = np.array([parse_timestamp_from_filename(p.name) for p in frames], dtype=np.float32)
    frame_names = np.array([p.name for p in frames])
    out_path = Path("data/embeddings/test1_visual.npz")
    save_visual_embeddings(out_path, embeddings, timestamps, frame_names)
    print("Final embeddings shape:", embeddings.shape)
    print("Final timestamps shape:", timestamps.shape)

if __name__ == "__main__":
    main()
