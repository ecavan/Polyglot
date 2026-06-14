import subprocess
import tempfile
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
    """Cumulative (start,end) per segment on the DUBBED clock (natural speed + gaps)."""
    gap = gap_ms / 1000.0
    timeline: list[tuple[float, float]] = []
    t = 0.0
    for seg in segments:
        dur = seg["audio_dur"]
        timeline.append((t, t + dur))
        t += dur + gap
    return timeline


def mix_speech_and_bed(speech: np.ndarray, bed: np.ndarray, bed_gain: float) -> np.ndarray:
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


def _rubberband(wav: np.ndarray, factor: float) -> np.ndarray:
    """Pitch-preserving tempo change (factor>1 = faster/shorter) via ffmpeg rubberband."""
    if abs(factor - 1.0) < 0.02 or len(wav) < 2048:
        return wav
    factor = max(0.5, min(2.0, factor))
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "i.wav", Path(td) / "o.wav"
        sf.write(str(src), wav, SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"rubberband=tempo={factor}",
                        "-ar", str(SR), "-ac", "1", str(dst)], check=True, capture_output=True)
        out, _ = sf.read(str(dst), dtype="float32")
    return np.asarray(out, dtype=np.float32)


def concat_audio(segments: list[dict], gap_ms: int) -> np.ndarray:
    """Dubbed-timeline track: segments back-to-back with gap_ms silence (podcasts)."""
    gap = np.zeros(int(gap_ms / 1000.0 * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    for seg in segments:
        data, _sr = sf.read(seg["audio_path"], dtype="float32")
        parts.append(data)
        parts.append(gap)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def build_synced_track(segments: list[dict], source_duration: float, speed: float = 1.1,
                       max_stretch: float = 1.8) -> np.ndarray:
    """Original-timeline track (video): place each French clip at its source start time and,
    when the French runs longer than its English window, SPEED IT UP (capped) to fit. Short
    lines keep their natural pace — the slot's leftover time stays as a pause, matching the
    original speaker. `speed` biases everything ~10% faster so the dub never drags."""
    total = max(1, int(source_duration * SR))
    track = np.zeros(total, dtype=np.float32)
    cursor = 0
    for seg in segments:
        wav, _sr = sf.read(seg["audio_path"], dtype="float32")
        wav = np.asarray(wav, dtype=np.float32)
        slot = max(0.0, seg["end"] - seg["start"])
        if slot > 0.05 and len(wav):
            factor = (len(wav) / SR) / slot * speed         # >1 = French overruns its slot
            if factor > 1.0:                                # only compress over-long lines
                wav = _rubberband(wav, min(max_stretch, factor))
        pos = max(int(seg["start"] * SR), cursor)           # anchor to source time; no overlap
        end = pos + len(wav)
        if end > len(track):
            track = np.concatenate([track, np.zeros(end - len(track), dtype=np.float32)])
        track[pos:end] += wav
        cursor = end
    return track


def _resample_bed(bed_path: Path, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-i", str(bed_path), "-ac", "1", "-ar", str(SR), str(out)],
                   check=True, capture_output=True)
    return out


def assemble(segments: list[dict], out_mp3: Path, settings: Settings,
            bed_path: Path | None = None, sync_to_source: bool = False,
            source_duration: float | None = None) -> EpisodeAudio:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if sync_to_source and source_duration:
        full = build_synced_track(segments, source_duration, speed=settings.video_speed)
        timeline = [(s["start"], s["end"]) for s in segments]   # subs align to source/video time
    else:
        full = concat_audio(segments, settings.gap_ms)
        timeline = build_timeline(segments, settings.gap_ms)

    if bed_path is not None and settings.mix_bed:
        bedwav = out_mp3.with_suffix(".bed.wav")
        if sync_to_source:
            _resample_bed(bed_path, bedwav)                      # original timing, no stretch
        else:
            _stretch_bed_to(bed_path, len(full) / SR, bedwav)    # stretch to dub length
        bed, _sr = sf.read(str(bedwav), dtype="float32")
        if bed.ndim > 1:
            bed = bed.mean(axis=1)
        full = mix_speech_and_bed(full, bed, settings.bed_gain)
        bedwav.unlink(missing_ok=True)

    tmp_wav = out_mp3.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), full, SR, subtype="FLOAT")
    try:  # loudness-normalize so the dub is comfortably audible
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:  # loudnorm can fail on near-silent audio
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            check=True,
        )
    tmp_wav.unlink(missing_ok=True)
    return EpisodeAudio(duration=len(full) / SR, byte_length=out_mp3.stat().st_size, timeline=timeline)


def _stretch_bed_to(bed_path: Path, target_seconds: float, out_path: Path) -> Path:
    dur = _audio_duration(bed_path)
    tempo = max(0.5, min(2.0, dur / target_seconds)) if target_seconds > 0 else 1.0
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(bed_path), "-filter:a", f"rubberband=tempo={tempo}",
         "-ac", "1", "-ar", str(SR), str(out_path)],
        check=True, capture_output=True,
    )
    return out_path
