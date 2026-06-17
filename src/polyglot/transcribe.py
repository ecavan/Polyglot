from pathlib import Path

from polyglot.config import Settings
from polyglot.segments import new_segment


def _tag_conf(seg: dict, avg_logprob, no_speech_prob) -> dict:
    """Carry whisper's confidence signals so we can drop garbled (quiet/noisy/accented) lines."""
    seg["avg_logprob"] = avg_logprob
    seg["no_speech_prob"] = no_speech_prob
    return seg


def segments_from_mlx_result(result: dict) -> list[dict]:
    return [
        _tag_conf(new_segment(index=i, start=s["start"], end=s["end"], text=s["text"].strip()),
                  s.get("avg_logprob"), s.get("no_speech_prob"))
        for i, s in enumerate(result["segments"])
    ]


def drop_low_confidence(segments: list[dict], min_logprob: float, max_no_speech: float) -> list[dict]:
    """Remove segments whisper wasn't confident about (low avg_logprob or high no-speech prob) —
    the garbled quiet/accented/background-noise speech that otherwise becomes wrong, out-of-sync
    French. Clear speech (narrator, podcast hosts) scores well and is kept. Re-indexes."""
    kept = []
    for s in segments:
        lp, ns = s.get("avg_logprob"), s.get("no_speech_prob")
        if (lp is not None and lp < min_logprob) or (ns is not None and ns > max_no_speech):
            continue
        kept.append(s)
    for i, s in enumerate(kept):
        s["index"] = i
    return kept


def _transcribe_mlx(wav: Path, settings: Settings) -> list[dict]:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=settings.mlx_whisper_repo,
        language="en",
    )
    return segments_from_mlx_result(result)


def _transcribe_faster(wav: Path, settings: Settings) -> list[dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(settings.faster_whisper, device="cpu", compute_type="int8")
    gen, _info = model.transcribe(str(wav), language="en", beam_size=5)
    return [
        _tag_conf(new_segment(index=i, start=s.start, end=s.end, text=s.text.strip()),
                  getattr(s, "avg_logprob", None), getattr(s, "no_speech_prob", None))
        for i, s in enumerate(gen)
    ]


def transcribe(wav: Path, settings: Settings) -> list[dict]:
    if settings.transcribe_backend == "mlx-whisper":
        return _transcribe_mlx(wav, settings)
    if settings.transcribe_backend == "faster-whisper":
        return _transcribe_faster(wav, settings)
    raise ValueError(f"unknown transcribe_backend: {settings.transcribe_backend}")
