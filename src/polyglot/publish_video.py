import subprocess
from pathlib import Path


def _duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def _escape_sub(path: Path) -> str:
    # ffmpeg subtitles filter: escape the few chars special inside the filtergraph.
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def mux(video_path: Path, audio_path: Path, out_mp4: Path, subtitle: Path | None = None) -> Path:
    """Replace the video's audio with the French dub and (optionally) BURN IN the bilingual
    subtitle so the FR+EN transcript is always on screen (any player). Pads the video by
    freezing the last frame when the dub is longer."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    delta = _duration(audio_path) - _duration(video_path)

    vf_parts = []
    if delta > 0.1:
        vf_parts.append(f"tpad=stop_mode=clone:stop_duration={delta:.3f}")
    if subtitle is not None:
        vf_parts.append(
            f"subtitles={_escape_sub(subtitle)}:force_style="
            "'Fontsize=15,Outline=1,Shadow=0,MarginV=18'"
        )

    if vf_parts:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-filter_complex", f"[0:v]{','.join(vf_parts)}[v]",
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
