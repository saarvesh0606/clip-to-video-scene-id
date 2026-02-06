from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INDEXED_DIR = ROOT / "data" / "indexed_videos"
FAISS_DIR = ROOT / "data" / "faiss"
INDEX_PATH = FAISS_DIR / "visual.index"
META_PATH = FAISS_DIR / "meta.json"


def run(cmd):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def clean_caches():
    """
    Remove cached frames / embeddings so deleted videos leave no disk artifacts.
    These are SAFE to delete; they will be regenerated on indexing.
    """
    CACHE_DIRS = [
        ROOT / "data" / "embeddings",
        ROOT / "data" / "embedding",
        ROOT / "data" / "index_frames",
        ROOT / "data" / "indexed_frames",
        ROOT / "data" / "frames",
        ROOT / "data" / "index_cache",
    ]

    for d in CACHE_DIRS:
        if d.exists() and d.is_dir():
            print(f"[CLEAN] Removing cache folder: {d}")
            shutil.rmtree(d, ignore_errors=True)

    # Remove stray .npz embedding cache files (safe)
    data_root = ROOT / "data"
    for npz in data_root.rglob("*.npz"):
        try:
            npz.unlink()
            print(f"[CLEAN] Removed cache file: {npz}")
        except Exception:
            pass


def main():
    if not INDEXED_DIR.exists():
        print(f"[ERR] Missing folder: {INDEXED_DIR}")
        sys.exit(1)

    vids = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"):
        vids.extend(sorted(INDEXED_DIR.glob(ext)))

    # Always ensure FAISS directory exists
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    if not vids:
        print(f"[WARN] No videos found in {INDEXED_DIR}")

        # Clear DB to avoid stale memory
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()

        # Clean caches as well
        clean_caches()

        print("[OK] Cleared FAISS index, meta, and caches (empty DB).")
        return

    # ---------------------------------------------------------
    # 1) Clear old FAISS DB (true deletion of old memory)
    # ---------------------------------------------------------
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if META_PATH.exists():
        META_PATH.unlink()

    # ---------------------------------------------------------
    # 2) Clean cache folders (OPTION B)
    # ---------------------------------------------------------
    clean_caches()

    # ---------------------------------------------------------
    # 3) Re-index everything present right now
    # ---------------------------------------------------------
    for v in vids:
        video_id = v.stem  # stable ID = filename stem
        run([
            "python",
            "indexing/index_video.py",
            "--video",
            str(v),
            "--video_id",
            video_id,
        ])

    print(f"[OK] Rebuilt index from {len(vids)} videos.")


if __name__ == "__main__":
    main()
