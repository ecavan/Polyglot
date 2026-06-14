import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from polyglot.config import JobSpec, Settings

SR = 24000  # XTTS output sample rate
CLONE = "__CLONE__"


def assign_voices(speakers: list[str], voice_pool: list[str], clone_available: bool) -> dict:
    """Map each distinct speaker label to a voice.

    With a cloned voice available, the first speaker uses the clone and the rest
    draw from the built-in pool. Otherwise every speaker draws from the pool
    (round-robin). Distinct speakers are ordered by label for stable assignment.
    """
    mapping: dict[str, str] = {}
    pool_i = 0
    for i, spk in enumerate(sorted(set(speakers))):
        if clone_available and i == 0:
            mapping[spk] = CLONE
        else:
            mapping[spk] = voice_pool[pool_i % len(voice_pool)]
            pool_i += 1
    return mapping


def synthesize_with(
    segments: list[dict],
    synth: Callable[[dict], np.ndarray],
    out_dir: Path,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        wav = np.asarray(synth(seg), dtype=np.float32)
        path = out_dir / f"seg_{seg['index']:04d}.wav"
        sf.write(str(path), wav, SR, subtype="FLOAT")
        seg["audio_path"] = str(path)
        seg["audio_dur"] = len(wav) / SR
    return segments


def _xtts_engine(segments: list[dict], job: JobSpec, settings: Settings) -> Callable[[dict], np.ndarray]:
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(settings.tts_device)
    model = tts.synthesizer.tts_model

    speakers = [seg.get("speaker") or "SPEAKER_00" for seg in segments]
    voice_for = assign_voices(speakers, settings.voice_pool, bool(job.voice_refs))

    # Compute conditioning latents once per distinct voice (expensive otherwise).
    latents: dict[str, tuple] = {}
    for vkey in set(voice_for.values()):
        if vkey == CLONE:
            gpt, spe = model.get_conditioning_latents(
                audio_path=[str(p) for p in job.voice_refs]
            )
        else:
            entry = model.speaker_manager.speakers[vkey]
            gpt, spe = entry["gpt_cond_latent"], entry["speaker_embedding"]
        latents[vkey] = (gpt, spe)

    params = dict(
        temperature=settings.tts_temperature,
        repetition_penalty=settings.tts_repetition_penalty,
        top_p=settings.tts_top_p,
        length_penalty=settings.tts_length_penalty,
        speed=settings.tts_speed,
        enable_text_splitting=False,
    )

    def synth(seg: dict) -> np.ndarray:
        spk = seg.get("speaker") or "SPEAKER_00"
        gpt, spe = latents[voice_for[spk]]
        out = model.inference(seg["translation"], job.target_lang, gpt, spe, **params)
        return np.asarray(out["wav"], dtype=np.float32)

    return synth


def synthesize(segments: list[dict], job: JobSpec, settings: Settings, out_dir: Path) -> list[dict]:
    if settings.tts_backend != "xtts":
        raise ValueError(f"unknown tts_backend: {settings.tts_backend}")
    synth = _xtts_engine(segments, job, settings)
    return synthesize_with(segments, synth, out_dir)
