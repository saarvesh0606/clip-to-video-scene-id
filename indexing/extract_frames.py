import os
import sys
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def safe_mkdir(path:Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def extract_frames(
    video_path:Path,
    output_dir:Path,
    fps_extract:float=1.0,
    max_frames:int=None,
    resize_width:int=None,
    overwrite:bool=False,
) -> None:
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    safe_mkdir(output_dir)

    
    existing = list(output_dir.glob("*.jpg"))
    if existing and not overwrite:
        print(f"Frames already exist in {output_dir}. Use --overwrite to re-extract.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        # fallback if FPS is not available
        src_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / src_fps if total_frames > 0 else None

    # How often (in frames) to sample
    # Example: src_fps=30, fps_extract=1 => take every 30 frames
    sample_every = max(int(round(src_fps / fps_extract)), 1)

    print("\n==============================")
    print("Frame Extraction Settings")
    print("==============================")
    print(f"Video:            {video_path.name}")
    print(f"Source FPS:       {src_fps:.3f}")
    print(f"Target Extract:   {fps_extract} FPS")
    print(f"Sample every:     {sample_every} frames")
    print(f"Total frames:     {total_frames}")
    if duration_sec:
        print(f"Duration:         {duration_sec:.2f} sec")
    print(f"Output dir:       {output_dir}")
    if resize_width:
        print(f"Resize width:     {resize_width}px (aspect preserved)")
    if max_frames:
        print(f"Max frames:       {max_frames}")
    print("==============================\n")

    saved = 0
    frame_idx = 0

    # Use tqdm with known total best-effort
    pbar_total = total_frames if total_frames > 0 else None
    pbar = tqdm(total=pbar_total, desc="Extracting", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            # Optional resize
            if resize_width is not None:
                h, w = frame.shape[:2]
                if w > 0 and w != resize_width:
                    new_h = int(h * (resize_width / w))
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

            # Timestamp in seconds (approx)
            t_sec = frame_idx / src_fps

            # Save name: videoName_t000012.34_f00001234.jpg
            out_name = f"{video_path.stem}_t{t_sec:010.2f}_f{frame_idx:09d}.jpg"
            out_path = output_dir / out_name

            cv2.imwrite(str(out_path), frame)
            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\nDone. Saved {saved} frames to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video at a target FPS.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Output directory to save frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Extraction FPS (default: 1.0)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to save (optional)")
    parser.add_argument("--resize_width", type=int, default=None, help="Resize frames to this width (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames in output folder")
    args = parser.parse_args()

    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.out),
        fps_extract=args.fps,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
    
    
    