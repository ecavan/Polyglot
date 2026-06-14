import subprocess
from pathlib import Path


def _duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def mux(video_path: Path, audio_path: Path, out_mp4: Path) -> Path:
    """Replace a video's audio with the French dub. If the dub is longer than the video
    (natural-speed dubs usually are), the video is padded by freezing the last frame so
    the durations match; otherwise the video stream is copied as-is."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    delta = _duration(audio_path) - _duration(video_path)
    if delta > 0.1:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={delta:.3f}[v]",
             "-map", "[v]", "-map", "1:a",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_mp4)],
            check=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_mp4)],
            check=True,
        )
    return out_mp4
