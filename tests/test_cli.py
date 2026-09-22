import json

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
