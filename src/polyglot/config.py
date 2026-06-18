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
    transcribe_min_logprob: float        # drop whisper segments below this avg_logprob (garbled)
    transcribe_max_no_speech: float      # ...or above this no_speech_prob
    translate_backend: str                # "gemini" | "claude" (remote, fall back to local) | "mlx"
    mlx_llm_repo: str
    anthropic_model: str
    gemini_model: str
    translate_chunk_size: int             # segments per Claude request (document-context batch)
    translate_context_lines: int          # prior lines shown read-only for cross-chunk continuity
    translate_max_retries: int
    tts_backend: str
    tts_device: str
    # "self" = clone each detected speaker from the episode audio; "pool" = built-in voices
    voice_mode: str
    # orpheus backend (expressive French TTS via llama.cpp GGUF + SNAC)
    orpheus_gguf: Path
    orpheus_voices: list[str]
    orpheus_voice_pitch: list
    orpheus_pitch_mode: str               # "fixed" (use voice_pitch) | "spread" | "random" (±1/±2)
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
    # local Jellyfin library + retention
    library_path: Path
    retention_keep: int
    retention_max_age_days: int
    # defaults
    clip_seconds: int
    max_video_minutes: int
    min_free_gb: float                   # worker stops before a job if free disk is below this
    min_episode_minutes: float           # skip items shorter than this (previews/trailers/shorts)
    video_speed: float
    video_max_stretch: float             # max speed-up of a dense line (low = slower/longer, clearer)
    video_height: int
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
    speakers: int | None = None     # expected speaker count (None = auto: youtube 1, podcast default)
    domain: str | None = None       # chess|poker|news|finance|sports|general -> gemini-audio prompt


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
    domain: str | None = None


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    m, p, df = d["models"], d["paths"], d["defaults"]
    tts = d.get("tts", {})
    sep = d.get("separation", {})
    mix = d.get("mix", {})
    orph = d.get("orpheus", {})
    lib = d.get("library", {})
    ret = d.get("retention", {})
    default_gguf = Path.home() / ".cache" / "polyglot" / "orpheus" / "Orpheus-3b-French-FT-Q8_0.gguf"
    return Settings(
        transcribe_backend=m["transcribe_backend"],
        mlx_whisper_repo=m["mlx_whisper_repo"],
        faster_whisper=m["faster_whisper"],
        transcribe_min_logprob=m.get("min_logprob", -1.0),
        transcribe_max_no_speech=m.get("max_no_speech", 0.6),
        translate_backend=m["translate_backend"],
        mlx_llm_repo=m["mlx_llm_repo"],
        anthropic_model=m.get("anthropic_model", "claude-sonnet-4-6"),
        gemini_model=m.get("gemini_model", "gemini-3.1-pro-preview"),
        translate_chunk_size=m.get("translate_chunk_size", 40),
        translate_context_lines=m.get("translate_context_lines", 3),
        translate_max_retries=m.get("translate_max_retries", 3),
        tts_backend=m["tts_backend"],
        tts_device=m["tts_device"],
        voice_mode=tts.get("voice_mode", "pool"),
        orpheus_gguf=Path(orph.get("gguf", str(default_gguf))),
        orpheus_voices=orph.get("voices", ["Pierre", "Pierre", "Pierre", "Pierre"]),
        orpheus_voice_pitch=orph.get("voice_pitch", [0, -1, -2, -3]),
        orpheus_pitch_mode=orph.get("pitch_mode", "fixed"),
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
        library_path=Path(lib.get("path", "~/PolyglotLibrary")).expanduser(),
        retention_keep=ret.get("keep", 10),
        retention_max_age_days=ret.get("max_age_days", 7),
        clip_seconds=df["clip_seconds"],
        max_video_minutes=df.get("max_video_minutes", 60),
        min_free_gb=df.get("min_free_gb", 3.0),
        min_episode_minutes=df.get("min_episode_minutes", 6),
        video_speed=df.get("video_speed", 1.0),
        video_max_stretch=df.get("video_max_stretch", 1.3),
        video_height=df.get("video_height", 720),
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
            speakers=s.get("speakers"),
            domain=s.get("domain"),
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
        domain=match.domain,
    )
