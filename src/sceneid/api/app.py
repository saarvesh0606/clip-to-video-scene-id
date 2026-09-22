"""HTTP API.

The model and library load once at startup. Matching is CPU-bound, so it runs in a worker
thread behind a semaphore: at most ``max_concurrent_matches`` run at once, and a request
that can't get a slot within ``queue_timeout_s`` gets a 503 instead of piling up.
"""

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..config import Settings
from ..frames import VideoError, probe
from ..library import LibraryError
from ..log import request_id_var
from ..matcher import Matcher
from ..matching import MatchResult
from .metrics import Metrics

log = logging.getLogger("sceneid.api")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MATCH_PATH = "/v1/match"


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(status_code=status, detail=message)
        self.code = code


class VideoSummary(BaseModel):
    video_id: str
    duration_s: float
    sample_fps: float
    n_vectors: int
    indexed_at: str


class VideoList(BaseModel):
    count: int
    videos: list[VideoSummary]


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}, "request_id": request_id_var.get()}


class RequestContextMiddleware:
    """Request ids, access logs, HTTP metrics and the upload size cap, as plain ASGI."""

    def __init__(self, app: ASGIApp, metrics: Metrics, max_upload_bytes: int):
        self.app = app
        self.metrics = metrics
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            if scope["path"] == _MATCH_PATH and scope["method"] == "POST":
                length = headers.get(b"content-length")
                if length and length.isdigit() and int(length) > self.max_upload_bytes:
                    response = JSONResponse(
                        _error_body("payload_too_large", self._too_large_message()), status_code=413
                    )
                    await response(scope, receive, send_wrapper)
                    return
                receive = self._limit_body(receive)
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            route_path = getattr(route, "path", None) or (
                _MATCH_PATH if scope["path"] == _MATCH_PATH else "unmatched"
            )
            self.metrics.http_requests.labels(scope["method"], route_path, str(status)).inc()
            self.metrics.http_latency.labels(scope["method"], route_path).observe(elapsed)
            level = logging.DEBUG if route_path in ("/metrics", "/health") else logging.INFO
            log.log(
                level,
                "http.request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round(elapsed * 1e3, 1),
                },
            )
            request_id_var.reset(token)

    def _too_large_message(self) -> str:
        return f"clip is larger than the {self.max_upload_bytes / 1e6:g} MB limit"

    def _limit_body(self, receive: Receive) -> Receive:
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_upload_bytes:
                    raise ApiError(413, "payload_too_large", self._too_large_message())
            return message

        return limited


def create_app(settings: Settings | None = None, matcher: Matcher | None = None) -> FastAPI:
    """Build the app. Pass `matcher` to skip loading the model (tests do this)."""
    settings = settings or Settings()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded = matcher or await anyio.to_thread.run_sync(Matcher.from_settings, settings)
        app.state.matcher = loaded
        # Created here, inside the running loop, not at import time.
        app.state.match_slots = asyncio.Semaphore(settings.max_concurrent_matches)
        metrics.library_videos.set(loaded.library.n_videos)
        metrics.library_vectors.set(loaded.library.n_vectors)
        log.info(
            "api.ready",
            extra={
                "version": __version__,
                "embedder": loaded.embedder.name,
                "algorithm": loaded.algorithm.name,
                "videos": loaded.library.n_videos,
                "vectors": loaded.library.n_vectors,
            },
        )
        yield

    app = FastAPI(
        title="sceneid",
        summary="Find which indexed video a short clip comes from, and where it starts.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    max_bytes = int(settings.max_upload_mb * 1e6)
    app.add_middleware(RequestContextMiddleware, metrics=metrics, max_upload_bytes=max_bytes)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = getattr(exc, "code", None) or f"http_{exc.status_code}"
        return JSONResponse(_error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        message = f"{where}: {first.get('msg', 'invalid request')}"
        return JSONResponse(_error_body("invalid_request", message), status_code=422)

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/health/ready", tags=["ops"])
    def ready(request: Request):
        """Readiness: the model and library are loaded."""
        loaded: Matcher | None = getattr(request.app.state, "matcher", None)
        if loaded is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        lib = loaded.library
        return {
            "status": "ready",
            "embedder": loaded.embedder.name,
            "algorithm": loaded.algorithm.name,
            "videos": lib.n_videos,
            "vectors": lib.n_vectors,
            "index_mb": round(lib.index_bytes / 1e6, 2),
        }

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/videos", tags=["library"])
    def list_videos(request: Request) -> VideoList:
        """The videos a clip can be matched against."""
        videos = request.app.state.matcher.library.videos()
        return VideoList(
            count=len(videos),
            videos=[
                VideoSummary(
                    video_id=v.video_id,
                    duration_s=v.duration_s,
                    sample_fps=v.sample_fps,
                    n_vectors=v.n_vectors,
                    indexed_at=v.indexed_at,
                )
                for v in videos
            ],
        )

    @app.post(
        _MATCH_PATH,
        tags=["match"],
        responses={
            413: {"description": "Clip larger than the upload limit"},
            415: {"description": "Not a supported video file type"},
            422: {"description": "Unreadable video, or longer than the clip limit"},
            503: {"description": "All matching slots are busy"},
        },
    )
    async def match(request: Request, clip: UploadFile = File(...)) -> MatchResult:
        """Identify which indexed video `clip` comes from, and where in it the clip starts.

        Returns `status: "unknown"` (not an error) when no video matches confidently.
        """
        loaded: Matcher = request.app.state.matcher
        slots: asyncio.Semaphore = request.app.state.match_slots
        suffix = Path(clip.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            metrics.matches.labels("bad_input").inc()
            allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
            raise ApiError(415, "unsupported_media_type", f"expected one of {allowed}")

        tmp_path = await anyio.to_thread.run_sync(_spool_to_disk, clip, suffix)
        try:
            metrics.upload_bytes.observe(tmp_path.stat().st_size)
            try:
                info = await anyio.to_thread.run_sync(probe, tmp_path)
            except VideoError:
                metrics.matches.labels("bad_input").inc()
                raise ApiError(
                    422, "unreadable_video", "the file is not a readable video"
                ) from None
            if info.duration_s > settings.max_clip_seconds:
                metrics.matches.labels("bad_input").inc()
                raise ApiError(
                    422,
                    "clip_too_long",
                    f"clip is {info.duration_s:.1f}s; the limit is {settings.max_clip_seconds:g}s",
                )

            try:
                await asyncio.wait_for(slots.acquire(), timeout=settings.queue_timeout_s)
            except asyncio.TimeoutError:
                metrics.matches.labels("busy").inc()
                raise ApiError(503, "busy", "all matching slots are busy; retry shortly") from None
            metrics.inflight.inc()
            try:
                result = await anyio.to_thread.run_sync(loaded.identify, tmp_path)
            except VideoError as exc:
                metrics.matches.labels("bad_input").inc()
                raise ApiError(422, "unreadable_video", str(exc)) from None
            except LibraryError as exc:
                metrics.matches.labels("error").inc()
                raise ApiError(503, "library_unavailable", str(exc)) from None
            except Exception:
                metrics.matches.labels("error").inc()
                log.exception("match.failed")
                raise ApiError(500, "internal_error", "matching failed") from None
            finally:
                metrics.inflight.dec()
                slots.release()
        finally:
            tmp_path.unlink(missing_ok=True)

        metrics.matches.labels(result.status).inc()
        metrics.confidence.observe(result.confidence)
        for stage in ("decode", "embed", "search", "decide"):
            if stage in result.timings_ms:
                metrics.stage_seconds.labels(stage).observe(result.timings_ms[stage] / 1e3)
        return result

    return app


def _spool_to_disk(upload: UploadFile, suffix: str) -> Path:
    """OpenCV needs a real path, so copy the upload to a named temp file."""
    fd, name = tempfile.mkstemp(prefix="sceneid-", suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, out, length=1 << 20)
    return Path(name)
