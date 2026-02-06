import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from query.match_local import match_video
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Clip-to-Video-ID", version="1.0")

class MatchPathRequest(BaseModel):
    video_path: str
    fps: float = 3.0
    max_frames: int = 40
    top_k: int = 10
    reextract: bool = False
    debug: bool = False
    min_conf: float = 0.80
    min_vote_ratio: float = 0.35


@app.post("/match_path")
def match_path(req: MatchPathRequest):
    p = Path(req.video_path)
    if not p.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"video_path not found: {req.video_path}"}
        )

    result = match_video(
        str(p),
        {
            "fps": req.fps,
            "max_frames": req.max_frames,
            "top_k": req.top_k,
            "reextract": req.reextract,
            "debug": req.debug,
            "json_only": True,
            "min_conf": req.min_conf,
            "min_vote_ratio": req.min_vote_ratio,
        },
    )
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
async def match_endpoint(
    video: UploadFile = File(...),
    fps: float = Query(3.0, ge=0.5, le=30.0),
    max_frames: int = Query(40, ge=1, le=500),
    top_k: int = Query(10, ge=1, le=50),
    reextract: bool = Query(False),
    debug: bool = Query(False),
    min_conf: float = Query(0.80, ge=0.0, le=1.0),
    min_vote_ratio: float = Query(0.35, ge=0.0, le=1.0),
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
                "json_only": True,
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
