from pathlib import Path

from polyglot.config import Settings
from polyglot.segments import new_segment


def segments_from_mlx_result(result: dict) -> list[dict]:
    return [
        new_segment(index=i, start=s["start"], end=s["end"], text=s["text"].strip())
        for i, s in enumerate(result["segments"])
    ]


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
        new_segment(index=i, start=s.start, end=s.end, text=s.text.strip())
        for i, s in enumerate(gen)
    ]


def transcribe(wav: Path, settings: Settings) -> list[dict]:
    if settings.transcribe_backend == "mlx-whisper":
        return _transcribe_mlx(wav, settings)
    if settings.transcribe_backend == "faster-whisper":
        return _transcribe_faster(wav, settings)
    raise ValueError(f"unknown transcribe_backend: {settings.transcribe_backend}")
