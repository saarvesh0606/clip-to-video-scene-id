import json
from dataclasses import replace
from pathlib import Path

import pytest

from sceneid.bench.download import DownloadError, download_film, require_films
from sceneid.bench.manifest import load_manifest
from tests.conftest import film_entry

REPO = Path(__file__).resolve().parents[1]


def test_download_verifies_and_unzips(mini_manifest, bench_workspace):
    films = require_films(bench_workspace, mini_manifest)
    assert films["film-a"].md5_verified and not films["film-b"].md5_verified
    assert films["film-b"].path.endswith("film-b.mp4")
    assert films["film-a"].duration_s == pytest.approx(90, abs=0.2)
    assert films["film-a"].pts_shift_s == 0.0
    # Downloading again is a no-op once a film is verified.
    again = download_film(mini_manifest.get("film-a"), bench_workspace)
    assert again.sha256 == films["film-a"].sha256


def test_checksum_mismatch_is_refused(mini_manifest, tmp_path):
    bad = replace(mini_manifest.get("film-a"), md5="0" * 32)
    with pytest.raises(DownloadError, match="md5"):
        download_film(bad, tmp_path, retries=1)
    assert not list((tmp_path / "films").glob("*.json"))


def test_missing_films_are_reported(mini_manifest, tmp_path):
    with pytest.raises(DownloadError, match="film-a, film-b, film-c"):
        require_films(tmp_path, mini_manifest)


def test_manifest_validation(tmp_path):
    base = film_entry("x", "library", "g", "file:///x", 1)
    for bad, message in [
        ([base, base], "unique"),
        ([{**base, "role": "maybe"}], "role"),
        ([{**base, "id": "has space"}], "valid video id"),
        ([{**base, "role": "heldout"}], "at least one library"),
        ([{**base, "surprise": 1}], "unknown fields"),
    ]:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"name": "m", "films": bad}))
        with pytest.raises(ValueError, match=message):
            load_manifest(path)


def test_tier1_manifest_is_valid():
    manifest = load_manifest(REPO / "benchmarks" / "datasets" / "tier1.json")
    assert len(manifest.library) == 13 and len(manifest.heldout) == 5
    assert all(f.license_url.startswith("https://creativecommons.org/") for f in manifest.films)
    # Every held-out film has an indexed look-alike in its group.
    library_groups = {f.group for f in manifest.library}
    assert all(f.group in library_groups for f in manifest.heldout)
