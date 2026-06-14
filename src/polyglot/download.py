import subprocess
from pathlib import Path

import requests


def ffmpeg_normalize_cmd(src: Path, dst: Path, clip_seconds: int) -> list[str]:
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", str(clip_seconds)]
    cmd += ["-ac", "1", "-ar", "16000", str(dst)]
    return cmd


def _download_to(url: str, dst: Path) -> Path:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dst


def fetch_audio(media_url: str, out_dir: Path, clip_seconds: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "source_raw"
    _download_to(media_url, raw)
    wav = out_dir / "source_16k_mono.wav"
    subprocess.run(ffmpeg_normalize_cmd(raw, wav, clip_seconds), check=True)
    return wav
