"""All tunable settings in one place.

Every field can be overridden with an environment variable prefixed ``SCENEID_``
(e.g. ``SCENEID_QUERY_FPS=2``) or from a ``.env`` file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCENEID_", env_file=".env", extra="ignore")

    # Library and model
    library_dir: Path = Path("data/library")
    embedder: str = "clip-vit-b32"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(32, ge=1, le=512)

    # Frame sampling. Frames are resized so their short side is `frame_short_side`
    # before embedding, which bounds memory without going below any model's input size.
    index_fps: float = Field(2.0, gt=0, le=30)
    query_fps: float = Field(3.0, gt=0, le=30)
    query_max_frames: int = Field(40, ge=1, le=1000)
    frame_short_side: int = Field(288, ge=32, le=2160)

    # Retrieval and decision
    top_k: int = Field(10, ge=1, le=200)
    # v1 stays the default until the benchmark shows v2 is better; see matching/v2.py.
    algorithm: Literal["v1", "v2"] = "v1"
    v1_min_conf: float = Field(0.83, ge=0, le=1)
    v1_min_vote_ratio: float = Field(0.90, ge=0, le=1)
    v2_min_score: float = Field(0.6, ge=0, le=1)
    v2_max_ratio: float = Field(0.9, ge=0, le=1)
    v2_tolerance_s: float = Field(0.6, gt=0, le=5)

    # API guardrails
    max_upload_mb: float = Field(50, gt=0)
    max_clip_seconds: float = Field(60, gt=0)
    max_concurrent_matches: int = Field(2, ge=1)
    queue_timeout_s: float = Field(30, gt=0)
    docs_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_json: bool = True
