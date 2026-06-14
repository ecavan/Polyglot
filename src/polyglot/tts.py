import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from polyglot.config import JobSpec, Settings

SR = 24000  # XTTS output sample rate


def synthesize_with(
    segments: list[dict],
    synth: Callable[[str], np.ndarray],
    out_dir: Path,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        wav = np.asarray(synth(seg["translation"]), dtype=np.float32)
        path = out_dir / f"seg_{seg['index']:04d}.wav"
        sf.write(str(path), wav, SR, subtype="FLOAT")
        seg["audio_path"] = str(path)
        seg["audio_dur"] = len(wav) / SR
    return segments


def _xtts_synth(job: JobSpec, settings: Settings) -> Callable[[str], np.ndarray]:
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(settings.tts_device)
    refs = [str(p) for p in job.voice_refs]

    def synth(text: str) -> np.ndarray:
        common = dict(text=text, language=job.target_lang, split_sentences=False)
        if refs:
            wav = tts.tts(**common, speaker_wav=refs)
        else:
            wav = tts.tts(**common, speaker="Claribel Dervla")
        return np.asarray(wav, dtype=np.float32)

    return synth


def synthesize(segments: list[dict], job: JobSpec, settings: Settings, out_dir: Path) -> list[dict]:
    if settings.tts_backend == "xtts":
        synth = _xtts_synth(job, settings)
    else:
        raise ValueError(f"unknown tts_backend: {settings.tts_backend}")
    return synthesize_with(segments, synth, out_dir)
