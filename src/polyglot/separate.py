from pathlib import Path

from polyglot.config import Settings


def stem_paths(audio_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Where Demucs writes the two stems for a given input track."""
    base = out_dir / "htdemucs" / Path(audio_path).stem
    return base / "vocals.wav", base / "no_vocals.wav"


def separate(audio_path: Path, out_dir: Path, settings: Settings) -> tuple[Path, Path]:
    """Split audio into (vocals, accompaniment) stems via Demucs --two-stems=vocals.

    Returns (vocals_path, no_vocals_path). CPU-only on Apple Silicon (~1.5x realtime).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals, no_vocals = stem_paths(audio_path, out_dir)
    if vocals.is_file() and no_vocals.is_file():
        return vocals, no_vocals  # already separated (idempotent)
    from demucs_infer.separate import main as demucs_main  # demucs-infer 4.1.x (torch 2.12 compatible)
    demucs_main([
        "--two-stems=vocals",
        "-d", settings.separate_device,
        "--segment", str(settings.separate_segment),
        "-o", str(out_dir),
        str(audio_path),
    ])
    return vocals, no_vocals
