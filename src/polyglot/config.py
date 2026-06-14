import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path("config/settings.toml")
DEFAULT_SHOWS_PATH = Path("config/shows.toml")


@dataclass
class Settings:
    # models
    transcribe_backend: str
    mlx_whisper_repo: str
    faster_whisper: str
    translate_backend: str
    mlx_llm_repo: str
    ollama_model: str
    ollama_url: str
    tts_backend: str
    tts_device: str
    # "self" = clone each detected speaker from the episode audio; "pool" = built-in voices
    voice_mode: str
    # orpheus backend (expressive French TTS via llama.cpp GGUF + SNAC)
    orpheus_gguf: Path
    orpheus_voices: list[str]
    orpheus_temperature: float
    orpheus_max_tokens: int
    # tts expressiveness (XTTS inference params) + multi-voice
    tts_temperature: float
    tts_repetition_penalty: float
    tts_top_p: float
    tts_length_penalty: float
    tts_speed: float
    voice_pool: list[str]
    # diarization
    num_speakers: int
    diarize_threshold: float
    # source separation + music-bed mixing
    separate_enabled: bool
    separate_device: str
    separate_segment: int
    mix_bed: bool
    bed_gain: float
    # paths
    cache_dir: Path
    output_dir: Path
    voices_dir: Path
    prompts_dir: Path
    state_path: Path
    # hosting
    hosting_type: str
    public_base_url: str
    bucket: str
    # defaults
    clip_seconds: int
    diarize: bool
    temperature: float
    max_tokens: int
    gap_ms: int


@dataclass
class ShowConfig:
    id: str
    title: str
    source: str
    source_type: str
    target_lang: str
    voice: str | None
    enabled: bool


@dataclass
class JobSpec:
    show_id: str
    title: str
    source: str
    source_type: str
    target_lang: str
    prompt_path: Path
    voice_refs: list[Path]
    settings: Settings


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    m, p, h, df = d["models"], d["paths"], d["hosting"], d["defaults"]
    tts = d.get("tts", {})
    sep = d.get("separation", {})
    mix = d.get("mix", {})
    orph = d.get("orpheus", {})
    default_gguf = Path.home() / ".cache" / "polyglot" / "orpheus" / "Orpheus-3b-French-FT-Q8_0.gguf"
    return Settings(
        transcribe_backend=m["transcribe_backend"],
        mlx_whisper_repo=m["mlx_whisper_repo"],
        faster_whisper=m["faster_whisper"],
        translate_backend=m["translate_backend"],
        mlx_llm_repo=m["mlx_llm_repo"],
        ollama_model=m["ollama_model"],
        ollama_url=m["ollama_url"],
        tts_backend=m["tts_backend"],
        tts_device=m["tts_device"],
        voice_mode=tts.get("voice_mode", "pool"),
        orpheus_gguf=Path(orph.get("gguf", str(default_gguf))),
        orpheus_voices=orph.get("voices", ["Pierre", "Amelie", "Marie"]),
        orpheus_temperature=orph.get("temperature", 0.7),
        orpheus_max_tokens=orph.get("max_tokens", 1800),
        tts_temperature=tts.get("temperature", 0.75),
        tts_repetition_penalty=tts.get("repetition_penalty", 6.0),
        tts_top_p=tts.get("top_p", 0.85),
        tts_length_penalty=tts.get("length_penalty", 1.0),
        tts_speed=tts.get("speed", 1.05),
        voice_pool=tts.get("voice_pool", ["Viktor Eka", "Andrew Chipper",
                                          "Craig Gutsy", "Damien Black"]),
        num_speakers=df.get("num_speakers", 2),
        diarize_threshold=df.get("diarize_threshold", 0.5),
        separate_enabled=sep.get("enabled", True),
        separate_device=sep.get("device", "cpu"),
        separate_segment=sep.get("segment", 7),
        mix_bed=mix.get("bed", True),
        bed_gain=mix.get("bed_gain", 0.6),
        cache_dir=Path(p["cache"]),
        output_dir=Path(p["output"]),
        voices_dir=Path(p["voices"]),
        prompts_dir=Path(p["prompts"]),
        state_path=Path(p["state"]),
        hosting_type=h["type"],
        public_base_url=h["public_base_url"],
        bucket=h["bucket"],
        clip_seconds=df["clip_seconds"],
        diarize=df["diarize"],
        temperature=df["temperature"],
        max_tokens=df["max_tokens"],
        gap_ms=df["gap_ms"],
    )


def load_shows(path: Path = DEFAULT_SHOWS_PATH) -> list[ShowConfig]:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    out: list[ShowConfig] = []
    for s in d.get("show", []):
        out.append(ShowConfig(
            id=s["id"],
            title=s["title"],
            source=s["source"],
            source_type=s["source_type"],
            target_lang=s["target_lang"],
            voice=s.get("voice"),
            enabled=s.get("enabled", True),
        ))
    return out


def _resolve_voice(voices_dir: Path, voice: str | None) -> list[Path]:
    if voice is None:
        return []
    wav = voices_dir / f"{voice}.wav"
    if wav.is_file():
        return [wav]
    folder = voices_dir / voice
    if folder.is_dir():
        clips = sorted(folder.glob("*.wav"))
        if clips:
            return clips
    raise FileNotFoundError(
        f"voice '{voice}' not found: expected {wav} or {folder}/*.wav"
    )


def build_job(
    show_id: str,
    settings: Settings | None = None,
    shows: list[ShowConfig] | None = None,
) -> JobSpec:
    settings = settings or load_settings()
    shows = shows if shows is not None else load_shows()
    match = next((s for s in shows if s.id == show_id), None)
    if match is None:
        raise KeyError(f"show '{show_id}' not found in shows.toml")

    prompt_path = settings.prompts_dir / f"{match.target_lang}.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"prompt for lang '{match.target_lang}' missing: {prompt_path}"
        )

    voice_refs = _resolve_voice(settings.voices_dir, match.voice)

    return JobSpec(
        show_id=match.id,
        title=match.title,
        source=match.source,
        source_type=match.source_type,
        target_lang=match.target_lang,
        prompt_path=prompt_path,
        voice_refs=voice_refs,
        settings=settings,
    )
