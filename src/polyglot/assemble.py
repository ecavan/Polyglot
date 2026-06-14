import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from polyglot.config import Settings
from polyglot.tts import SR


@dataclass
class EpisodeAudio:
    duration: float
    byte_length: int
    timeline: list[tuple[float, float]]


def build_timeline(segments: list[dict], gap_ms: int) -> list[tuple[float, float]]:
    gap = gap_ms / 1000.0
    timeline: list[tuple[float, float]] = []
    t = 0.0
    for seg in segments:
        dur = seg["audio_dur"]
        timeline.append((t, t + dur))
        t += dur + gap
    return timeline


def concat_audio(segments: list[dict], gap_ms: int) -> np.ndarray:
    gap = np.zeros(int(gap_ms / 1000.0 * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    for seg in segments:
        data, _sr = sf.read(seg["audio_path"], dtype="float32")
        parts.append(data)
        parts.append(gap)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def assemble(segments: list[dict], out_mp3: Path, settings: Settings) -> EpisodeAudio:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    full = concat_audio(segments, settings.gap_ms)
    tmp_wav = out_mp3.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), full, SR, subtype="FLOAT")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
        check=True,
    )
    tmp_wav.unlink(missing_ok=True)
    return EpisodeAudio(
        duration=len(full) / SR,
        byte_length=out_mp3.stat().st_size,
        timeline=build_timeline(segments, settings.gap_ms),
    )
