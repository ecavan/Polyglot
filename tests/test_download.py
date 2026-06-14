from pathlib import Path

from polyglot.download import ffmpeg_normalize_cmd, build_ydl_opts


def test_ffmpeg_cmd_full_episode():
    cmd = ffmpeg_normalize_cmd(Path("in.mp3"), Path("out.wav"), clip_seconds=0)
    assert cmd[0] == "ffmpeg"
    assert "-t" not in cmd
    assert "-ac" in cmd and "2" in cmd     # stereo for Demucs
    assert "44100" in cmd
    assert cmd[-1] == "out.wav"


def test_ffmpeg_cmd_trimmed():
    cmd = ffmpeg_normalize_cmd(Path("in.mp3"), Path("out.wav"), clip_seconds=180)
    i = cmd.index("-t")
    assert cmd[i + 1] == "180"


def test_build_ydl_opts_full():
    opts = build_ydl_opts(Path("/work"), clip_seconds=0)
    assert opts["merge_output_format"] == "mp4"
    assert "download_ranges" not in opts
    assert opts["outtmpl"].endswith("video.%(ext)s")


def test_build_ydl_opts_clipped():
    opts = build_ydl_opts(Path("/work"), clip_seconds=120)
    assert "download_ranges" in opts          # only first 120s downloaded
    assert opts["force_keyframes_at_cuts"] is True
