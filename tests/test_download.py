from pathlib import Path

from polyglot.download import ffmpeg_normalize_cmd


def test_ffmpeg_cmd_full_episode():
    cmd = ffmpeg_normalize_cmd(Path("in.mp3"), Path("out.wav"), clip_seconds=0)
    assert cmd[0] == "ffmpeg"
    assert "-t" not in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "16000" in cmd
    assert cmd[-1] == "out.wav"


def test_ffmpeg_cmd_trimmed():
    cmd = ffmpeg_normalize_cmd(Path("in.mp3"), Path("out.wav"), clip_seconds=180)
    i = cmd.index("-t")
    assert cmd[i + 1] == "180"
