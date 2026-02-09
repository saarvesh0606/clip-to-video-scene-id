import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import single source of truth defaults + matcher
from query.match_local import match_video, DEFAULT_PARAMS

app = FastAPI(title="Clip-to-Video-ID", version="1.0")


class MatchPathRequest(BaseModel):
    video_path: str

    # Defaults come from DEFAULT_PARAMS (single source of truth)
    fps: float = DEFAULT_PARAMS["fps"]
    max_frames: int = DEFAULT_PARAMS["max_frames"]
    top_k: int = DEFAULT_PARAMS["top_k"]

    reextract: bool = False
    debug: bool = False

    min_conf: float = DEFAULT_PARAMS["min_conf"]
    min_vote_ratio: float = DEFAULT_PARAMS["min_vote_ratio"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match_path")
def match_path(req: MatchPathRequest):
    p = Path(req.video_path)
    if not p.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"video_path not found: {req.video_path}"},
        )

    # Pass overrides only (still explicit so it's clear)
    result = match_video(
        str(p),
        {
            "fps": req.fps,
            "max_frames": req.max_frames,
            "top_k": req.top_k,
            "reextract": req.reextract,
            "debug": req.debug,
            "json_only": True,  # API output should always be JSON-only
            "min_conf": req.min_conf,
            "min_vote_ratio": req.min_vote_ratio,
        },
    )
    return JSONResponse(content=result)


@app.post("/match")
async def match_endpoint(
    video: UploadFile = File(...),

    # Swagger defaults pulled from DEFAULT_PARAMS (single source of truth)
    fps: float = Query(DEFAULT_PARAMS["fps"], ge=0.5, le=30.0),
    max_frames: int = Query(DEFAULT_PARAMS["max_frames"], ge=1, le=500),
    top_k: int = Query(DEFAULT_PARAMS["top_k"], ge=1, le=50),

    reextract: bool = Query(False),
    debug: bool = Query(False),

    min_conf: float = Query(DEFAULT_PARAMS["min_conf"], ge=0.0, le=1.0),
    min_vote_ratio: float = Query(DEFAULT_PARAMS["min_vote_ratio"], ge=0.0, le=1.0),
):
    # Save upload to temp file
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await video.read())

    try:
        result = match_video(
            tmp_path,
            {
                "fps": fps,
                "max_frames": max_frames,
                "top_k": top_k,
                "reextract": reextract,
                "debug": debug,
                "json_only": True,  # API output should always be JSON-only
                "min_conf": min_conf,
                "min_vote_ratio": min_vote_ratio,
            },
        )
        return JSONResponse(content=result)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
