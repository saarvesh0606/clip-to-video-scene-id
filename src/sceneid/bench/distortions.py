"""The ways a query clip is altered, modelled on how clips really get re-shared.

Each distortion is an ffmpeg filter chain plus encoder settings. Random parameters (crop
position, colour shift, tilt...) come from the query's own seed, so re-rendering a query
always gives the same clip.
"""

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Every chain ends here: even dimensions (x264 needs them) and a standard pixel format.
_NORMALIZE = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"


class RenderError(Exception):
    pass


@dataclass(frozen=True)
class Distortion:
    name: str
    description: str
    chain: Callable[[np.random.Generator], str]
    crf: int = 23
    speed: float = 1.0
    overlay: bool = False


def _crop(rng: np.random.Generator) -> str:
    keep = rng.uniform(0.6, 0.8)
    x, y = rng.uniform(0, 1 - keep, size=2)
    return f"crop=iw*{keep:.3f}:ih*{keep:.3f}:iw*{x:.3f}:ih*{y:.3f}"


def _color(rng: np.random.Generator) -> str:
    brightness = rng.choice([-1, 1]) * rng.uniform(0.04, 0.10)
    contrast = rng.uniform(1.15, 1.4)
    saturation = rng.uniform(1.3, 1.7)
    hue = rng.choice([-1, 1]) * rng.uniform(15, 35)
    return (
        f"eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f},"
        f"hue=h={hue:.1f}"
    )


def _screen_recording(rng: np.random.Generator) -> str:
    angle = rng.choice([-1, 1]) * rng.uniform(0.015, 0.04)  # radians, about 1-2 degrees
    return (
        "scale=iw*0.88:-2,"
        "pad=iw/0.88:ih/0.88:(ow-iw)/2:(oh-ih)/2:color=0x202020,"
        f"rotate={angle:.4f}:fillcolor=0x101010,"
        "gblur=sigma=1.2,noise=alls=10:allf=t,eq=gamma=1.15:saturation=0.85"
    )


_DISTORTIONS = [
    Distortion(
        "original", "Re-encoded at high quality; nothing else changed", lambda r: "null", crf=18
    ),
    Distortion("compression", "Heavy compression (x264 CRF 38)", lambda r: "null", crf=38),
    Distortion("downscale", "Scaled down to 240 pixels tall", lambda r: "scale=-2:240", crf=26),
    Distortion("crop", "Random crop keeping 60-80% of the width and height", _crop),
    Distortion(
        "letterbox",
        "Padded to a square with black bars, as reposted to social media",
        lambda r: "pad=max(iw\\,ih):max(iw\\,ih):(ow-iw)/2:(oh-ih)/2,scale=-2:'min(ih,1080)'",
    ),
    Distortion("mirror", "Flipped left to right", lambda r: "hflip"),
    Distortion("color", "Brightness, contrast, saturation and hue shifted", _color),
    Distortion(
        "overlay",
        "A channel logo and a news-style caption bar burned in",
        lambda r: "null",
        overlay=True,
    ),
    Distortion("speed", "Played 1.25x faster", lambda r: "setpts=PTS/1.25", speed=1.25),
    Distortion(
        "screen_recording",
        "Filmed off a screen: bezel, 1-2 degree tilt, blur, sensor noise, gamma shift",
        _screen_recording,
        crf=30,
    ),
]
DISTORTIONS: dict[str, Distortion] = {d.name: d for d in _DISTORTIONS}

_CAPTIONS = ["BREAKING NEWS", "LIVE", "EXCLUSIVE CLIP", "WATCH NOW", "TRENDING"]


def write_overlay(path: Path, width: int, height: int, rng: np.random.Generator) -> None:
    """A transparent PNG with a caption bar along the bottom and a logo in a top corner."""
    img = np.zeros((height, width, 4), np.uint8)
    bar = int(height * 0.12)
    img[height - bar :, :] = (20, 20, 160, 200)  # BGRA: dark red, mostly opaque
    scale = height / 400
    caption = str(rng.choice(_CAPTIONS))
    cv2.putText(
        img,
        caption,
        (int(width * 0.03), height - int(bar * 0.3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1 * scale,
        (255, 255, 255, 255),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )
    logo_w, logo_h = int(width * 0.14), int(height * 0.10)
    x0 = int(width * 0.03) if rng.random() < 0.5 else width - logo_w - int(width * 0.03)
    y0 = int(height * 0.04)
    img[y0 : y0 + logo_h, x0 : x0 + logo_w] = (255, 255, 255, 170)
    cv2.putText(
        img,
        f"CH {int(rng.integers(2, 99))}",
        (x0 + int(logo_w * 0.12), y0 + int(logo_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * scale,
        (30, 30, 30, 255),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), img):
        raise RenderError(f"could not write overlay image {path}")


def _capped(width: int, height: int) -> tuple[int, int]:
    if height <= 720:
        return width, height
    return int(round(width * 720 / height / 2) * 2), 720


def render(
    distortion: str,
    source: str | Path,
    start_s: float,
    duration_s: float,
    out: Path,
    *,
    seed: int,
    width: int,
    height: int,
) -> None:
    """Cut `duration_s` seconds of `source` from `start_s`, apply the distortion, write `out`."""
    if shutil.which("ffmpeg") is None:
        raise RenderError("ffmpeg is not installed")
    d = DISTORTIONS[distortion]
    rng = np.random.default_rng(seed)
    # Clips are rendered at most 720p tall, as re-shared clips usually are; it also halves
    # render time on 1080p sources.
    chain = f"scale=-2:'min(ih,720)',{d.chain(rng)}"
    if d.overlay:
        width, height = _capped(width, height)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        str(source),
    ]
    overlay_png = out.with_suffix(".overlay.png")
    if d.overlay:
        write_overlay(overlay_png, width, height, rng)
        cmd += [
            "-i",
            str(overlay_png),
            "-filter_complex",
            f"[0:v:0]{chain}[v];[v][1:v]overlay=0:0,{_NORMALIZE}[out]",
            "-map",
            "[out]",
        ]
    else:
        cmd += ["-map", "0:v:0", "-vf", f"{chain},{_NORMALIZE}"]
    cmd += ["-an", "-sn", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(d.crf), str(out)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    finally:
        overlay_png.unlink(missing_ok=True)
    if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        raise RenderError(f"ffmpeg failed for {distortion}: {' | '.join(tail) or 'no output'}")
