import subprocess
from pathlib import Path

import requests


def ffmpeg_normalize_cmd(src: Path, dst: Path, clip_seconds: int) -> list[str]:
    # Full-quality stereo 44.1 kHz — what Demucs wants for separation.
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", str(clip_seconds)]
    cmd += ["-ac", "2", "-ar", "44100", str(dst)]
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
    wav = out_dir / "source_44k.wav"
    subprocess.run(ffmpeg_normalize_cmd(raw, wav, clip_seconds), check=True)
    return wav


def to_16k_mono(src: Path, out_dir: Path) -> Path:
    """Downmix to 16 kHz mono — what Whisper and the diarizer expect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "speech_16k_mono.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)], check=True)
    return out
