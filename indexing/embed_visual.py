from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
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


def main():
    frames_dir = Path("data/embeddings/test1_frames")
    frames = list_frames(frames_dir)

    print(f"\nFound {len(frames)} frames in: {frames_dir}\n")
    for f in frames[:30]:
        ts = parse_timestamp_from_filename(f.name)
        print(f"{f.name} -> {ts}")

    print("\n✅ Timestamp parsing + frame listing works.\n")


if __name__ == "__main__":
    main()
