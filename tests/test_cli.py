import json
from pathlib import Path

import pytest

from sceneid.cli import main


def test_index_list_match_remove(tmp_path, videos, capsys):
    lib = ["--library", str(tmp_path / "lib"), "--embedder", "tiny16"]

    assert main([*lib, "index", str(videos["ref_a"]), str(videos["ref_b"])]) == 0
    assert "library: 2 videos, 120 vectors" in capsys.readouterr().out

    assert main([*lib, "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["embedder"] == "tiny16"
    assert [v["video_id"] for v in listing["videos"]] == ["ref_a", "ref_b"]

    assert main([*lib, "match", str(videos["clip_a"]), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "match" and result["video_id"] == "ref_a"

    assert main([*lib, "remove", "ref_b"]) == 0
    assert main([*lib, "remove", "ref_b"]) == 1
    assert "not in the library" in capsys.readouterr().err


def test_index_reports_failures_but_keeps_going(tmp_path, videos, capsys):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    lib = ["--library", str(tmp_path / "lib"), "--embedder", "tiny16"]
    assert main([*lib, "index", str(bad), str(videos["ref_a"])]) == 1
    assert "library: 1 videos" in capsys.readouterr().out


def test_bench_command_gets_its_own_options(tmp_path, capsys):
    manifest = str(Path(__file__).resolve().parents[1] / "benchmarks" / "datasets" / "tier1.json")
    bench = ["--embedder", "tiny16", "bench", "--manifest", manifest, "--workspace", str(tmp_path)]
    assert main([*bench, "status"]) == 0
    out = capsys.readouterr().out
    assert "films downloaded: 0/18" in out and "answer key: not built" in out

    with pytest.raises(SystemExit) as exit_info:
        main(["bench", "--help"])  # reaches the bench parser, not the top-level one
    assert exit_info.value.code == 0
    assert "evaluate" in capsys.readouterr().out
