import subprocess
from pathlib import Path


def _duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def _escape_sub(path: Path) -> str:
    # ffmpeg subtitles/ass filter: escape chars special inside the filtergraph. Commas and
    # brackets matter too — the library basename now contains "[ep_id]".
    out = str(path)
    for ch in ("\\", ":", "'", ",", "[", "]"):
        out = out.replace(ch, "\\" + ch)
    return out


def make_audio_video(audio_path: Path, subtitle: Path, out_mp4: Path,
                     bg: str = "0x111418") -> Path:
    """Render a podcast MP3 into a minimal MP4 for the TV: a static dark background
    with the styled side-by-side FR/EN transcript burned in, so Jellyfin on the Roku
    always shows the transcript (its audio-only subtitle support is unreliable)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    dur = _duration(audio_path)
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c={bg}:s=1920x1080:r=10:d={dur:.3f}",
         "-i", str(audio_path),
         "-vf", f"ass={_escape_sub(subtitle)}",
         "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_mp4)],
        check=True,
    )
    return out_mp4


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
        if str(subtitle).endswith(".ass"):
            vf_parts.append(f"ass={_escape_sub(subtitle)}")          # styled side-by-side (libass)
        else:
            vf_parts.append(f"subtitles={_escape_sub(subtitle)}:force_style="
                            "'Fontsize=15,Outline=1,Shadow=0,MarginV=18'")

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
