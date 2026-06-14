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


def mix_speech_and_bed(speech: np.ndarray, bed: np.ndarray, bed_gain: float) -> np.ndarray:
    """Overlay a (ducked) music bed under the speech. Bed is padded/truncated to the
    speech length; the sum is clamped to avoid clipping."""
    n = len(speech)
    if len(bed) < n:
        bed = np.pad(bed, (0, n - len(bed)))
    else:
        bed = bed[:n]
    mixed = speech + bed_gain * bed
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def _audio_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def _stretch_bed_to(bed_path: Path, target_seconds: float, out_path: Path) -> Path:
    """Pitch-preserving stretch of the music bed to ~target_seconds, mono @ SR.
    tempo<1 lengthens; clamped to rubberband's safe range."""
    dur = _audio_duration(bed_path)
    tempo = max(0.5, min(2.0, dur / target_seconds)) if target_seconds > 0 else 1.0
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(bed_path), "-filter:a", f"rubberband=tempo={tempo}",
         "-ac", "1", "-ar", str(SR), str(out_path)],
        check=True,
    )
    return out_path


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


def assemble(segments: list[dict], out_mp3: Path, settings: Settings,
            bed_path: Path | None = None) -> EpisodeAudio:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    full = concat_audio(segments, settings.gap_ms)
    if bed_path is not None and settings.mix_bed:
        stretched = out_mp3.with_suffix(".bed.wav")
        _stretch_bed_to(bed_path, len(full) / SR, stretched)
        bed, _sr = sf.read(str(stretched), dtype="float32")
        if bed.ndim > 1:
            bed = bed.mean(axis=1)
        full = mix_speech_and_bed(full, bed, settings.bed_gain)
        stretched.unlink(missing_ok=True)
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
