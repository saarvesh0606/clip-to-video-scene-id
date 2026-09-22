import pytest
from fastapi.testclient import TestClient

from sceneid.api import create_app
from sceneid.config import Settings


def _client(matcher, **settings) -> TestClient:
    return TestClient(create_app(Settings(**settings), matcher=matcher))


@pytest.fixture
def client(matcher):
    with _client(matcher) as c:
        yield c


def _upload(client, path, name=None):
    with open(path, "rb") as f:
        return client.post("/v1/match", files={"clip": (name or path.name, f, "video/mp4")})


def test_health_and_readiness(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/health/ready").json()
    assert ready["status"] == "ready" and ready["videos"] == 2 and ready["embedder"] == "tiny16"


def test_lists_videos(client):
    body = client.get("/v1/videos").json()
    assert body["count"] == 2
    assert [v["video_id"] for v in body["videos"]] == ["ref_a", "ref_b"]
    assert "source_path" not in body["videos"][0]  # server paths stay private


def test_match(client, videos):
    r = _upload(client, videos["clip_a"])
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "match" and body["video_id"] == "ref_a"
    assert body["offset_s"] == pytest.approx(11.0, abs=0.5)
    assert r.headers["x-request-id"]


def test_unknown_is_a_200_not_an_error(client, videos):
    r = _upload(client, videos["unknown"])
    assert r.status_code == 200 and r.json()["status"] == "unknown"


def test_request_id_is_propagated(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["x-request-id"] != "bad id with spaces"


def test_rejects_non_video_extensions(client, videos):
    r = _upload(client, videos["clip_a"], name="clip.txt")
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_rejects_unreadable_video(client, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 5000)
    r = _upload(client, bad)
    assert r.status_code == 422 and r.json()["error"]["code"] == "unreadable_video"


def test_rejects_oversized_upload(matcher, videos):
    with _client(matcher, max_upload_mb=0.001) as c:
        r = _upload(c, videos["clip_a"])
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_rejects_long_clips(matcher, videos):
    with _client(matcher, max_clip_seconds=2) as c:
        r = _upload(c, videos["clip_a"])
    assert r.status_code == 422 and r.json()["error"]["code"] == "clip_too_long"


def test_missing_file_field(client):
    r = client.post("/v1/match")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_unknown_route_uses_the_error_format(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_404"


def test_metrics_count_outcomes_and_routes(client, videos):
    _upload(client, videos["clip_a"])
    _upload(client, videos["unknown"])
    text = client.get("/metrics").text
    assert 'sceneid_matches_total{outcome="match"} 1.0' in text
    assert 'sceneid_matches_total{outcome="unknown"} 1.0' in text
    assert 'route="/v1/match",status="200"' in text
    assert "sceneid_match_stage_seconds_bucket" in text
    assert "sceneid_library_vectors 120.0" in text
