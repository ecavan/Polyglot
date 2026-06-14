import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from polyglot.config import JobSpec, Settings

SR = 24000  # XTTS output sample rate
CLONE = "__CLONE__"


def assign_voices(speakers: list[str], voice_pool: list[str], clone_available: bool) -> dict:
    """Map each distinct speaker label to a built-in pool voice (pool mode).

    With a user-provided clone available, the first speaker uses the clone and the
    rest draw from the pool. Distinct speakers are ordered by label for stability.
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


def select_reference_spans(segments: list[dict], target_seconds: float = 15.0) -> dict:
    """Per speaker, pick the SINGLE longest contiguous segment as the cloning reference.

    A clean, contiguous reference clones far better than several stitched-together
    segments (the abrupt joins make XTTS hallucinate). Returns {speaker: [segment]}.
    """
    by_spk: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg.get("speaker") or "SPEAKER_00"
        by_spk.setdefault(spk, []).append(seg)
    return {spk: [max(segs, key=lambda s: s["end"] - s["start"])] for spk, segs in by_spk.items()}


def build_speaker_references(source_wav: Path, segments: list[dict], out_dir: Path,
                            target_seconds: float = 15.0, min_ref: float = 8.0) -> dict:
    """Extract a per-speaker reference clip (single longest segment, peak-normalized)
    from the source audio. Speakers whose longest contiguous segment is shorter than
    `min_ref` seconds are omitted — XTTS cloning from sub-~8s references is unstable
    (rambling/hallucination), so the caller falls back to a stable built-in voice."""
    audio, sr = sf.read(str(source_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Path] = {}
    for spk, segs in select_reference_spans(segments, target_seconds).items():
        seg = segs[0]
        if (seg["end"] - seg["start"]) < min_ref:
            continue
        clip = audio[int(seg["start"] * sr):int(seg["end"] * sr)].copy()
        peak = float(np.max(np.abs(clip))) if clip.size else 0.0
        if peak > 0:
            clip = clip / peak * 0.95
        path = out_dir / f"ref_{spk}.wav"
        sf.write(str(path), clip, sr, subtype="FLOAT")
        refs[spk] = path
    return refs


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


def _xtts_engine(segments, job, settings, out_dir, source_wav=None) -> Callable[[dict], np.ndarray]:
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(settings.tts_device)
    model = tts.synthesizer.tts_model

    speakers = [seg.get("speaker") or "SPEAKER_00" for seg in segments]
    uniq = sorted(set(speakers))

    # "self" mode: clone each detected speaker from their own episode audio.
    refs: dict[str, Path] = {}
    if settings.voice_mode == "self" and source_wav is not None:
        refs = build_speaker_references(source_wav, segments, out_dir / "refs")

    # Build conditioning latents once per speaker.
    latents: dict[str, tuple] = {}
    pool_i = 0
    for i, spk in enumerate(uniq):
        if spk in refs:                                            # self-clone
            gpt, spe = model.get_conditioning_latents(audio_path=[str(refs[spk])])
        elif settings.voice_mode != "self" and job.voice_refs and i == 0:  # user clip
            gpt, spe = model.get_conditioning_latents(
                audio_path=[str(p) for p in job.voice_refs]
            )
        else:                                                      # built-in pool fallback
            name = settings.voice_pool[pool_i % len(settings.voice_pool)]
            pool_i += 1
            entry = model.speaker_manager.speakers[name]
            gpt, spe = entry["gpt_cond_latent"], entry["speaker_embedding"]
        latents[spk] = (gpt, spe)

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
        gpt, spe = latents[spk]
        out = model.inference(seg["translation"], job.target_lang, gpt, spe, **params)
        return np.asarray(out["wav"], dtype=np.float32)

    return synth


def synthesize(segments, job, settings, out_dir, source_wav=None) -> list[dict]:
    if settings.tts_backend != "xtts":
        raise ValueError(f"unknown tts_backend: {settings.tts_backend}")
    synth = _xtts_engine(segments, job, settings, out_dir, source_wav)
    return synthesize_with(segments, synth, out_dir)
