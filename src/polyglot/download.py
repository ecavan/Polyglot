import subprocess
from pathlib import Path

import requests

# Some podcast CDNs (e.g. Acast) 403 the default "python-requests" User-Agent as a bot.
# Send a normal browser UA so downloads behave like any podcast client.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}


def ffmpeg_normalize_cmd(src: Path, dst: Path, clip_seconds: int) -> list[str]:
    # Full-quality stereo 44.1 kHz — what Demucs wants for separation.
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", str(clip_seconds)]
    cmd += ["-ac", "2", "-ar", "44100", str(dst)]
    return cmd


def _download_to(url: str, dst: Path) -> Path:
    with requests.get(url, stream=True, timeout=120, headers=_HEADERS) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dst


def fetch_audio(media_url: str, out_dir: Path, clip_seconds: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "source_raw"
    _download_to(media_url, raw)
    wav = out_dir / "source_44k.wav"
    subprocess.run(ffmpeg_normalize_cmd(raw, wav, clip_seconds), check=True)
    return wav


def to_16k_mono(src: Path, out_dir: Path) -> Path:
    """Downmix to 16 kHz mono — what Whisper and the diarizer expect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "speech_16k_mono.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)], check=True)
    return out


def build_ydl_opts(out_dir: Path, clip_seconds: int) -> dict:
    opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    if clip_seconds and clip_seconds > 0:
        from yt_dlp.utils import download_range_func
        opts["download_ranges"] = download_range_func(None, [(0, clip_seconds)])
        opts["force_keyframes_at_cuts"] = True
    return opts


def video_metadata(url: str) -> dict:
    """Quick metadata (no download) for a YouTube URL — id, title, channel, duration (sec)."""
    from yt_dlp import YoutubeDL
    from polyglot.feeds import _yyyymmdd_to_epoch
    with YoutubeDL({"quiet": True, "noprogress": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "video_id": info.get("id", ""),
        "title": info.get("title", "(video)"),
        "channel": info.get("channel") or info.get("uploader") or "YouTube",
        "duration": info.get("duration") or 0,
        "published_ts": _yyyymmdd_to_epoch(info.get("upload_date")),
    }


def fetch_video(url: str, out_dir: Path, clip_seconds: int = 0, max_minutes: int = 60) -> Path:
    """Download a YouTube video (video+audio merged to mp4). Rejects videos longer
    than max_minutes. clip_seconds>0 downloads only the first N seconds (for testing)."""
    from yt_dlp import YoutubeDL
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = build_ydl_opts(out_dir, clip_seconds)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        dur = info.get("duration") or 0
        if max_minutes and dur > max_minutes * 60:
            raise ValueError(f"video is {dur/60:.0f} min (> max_video_minutes={max_minutes})")
        ydl.download([url])
    vids = [p for p in sorted(out_dir.glob("video.*")) if p.suffix in (".mp4", ".mkv", ".webm")]
    if not vids:
        raise FileNotFoundError("yt-dlp produced no video file")
    return vids[0]


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Pull a full-quality 44.1 kHz stereo audio track out of a video (for Demucs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "source_44k.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "2", "-ar", "44100", str(out)], check=True)
    return out
