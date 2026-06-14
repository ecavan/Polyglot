from pathlib import Path

import pytest

from polyglot.config import build_job, load_settings, load_shows, Settings

SETTINGS_TOML = """
[models]
transcribe_backend = "mlx-whisper"
mlx_whisper_repo   = "mlx-community/whisper-large-v3-turbo"
faster_whisper     = "medium.en"
translate_backend  = "mlx"
mlx_llm_repo       = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ollama_model       = "mistral"
ollama_url         = "http://localhost:11434/api/chat"
tts_backend        = "xtts"
tts_device         = "cpu"

[paths]
cache = "cache"
output = "output"
voices = "voices"
prompts = "config/prompts"
state = "state/processed.json"

[hosting]
type = "r2"
public_base_url = "https://media.example.com"
bucket = "polyglot-media"

[defaults]
clip_seconds = 0
diarize = false
temperature = 0.3
max_tokens = 512
gap_ms = 200
"""

SHOWS_TOML = """
[[show]]
id = "pti-fr"
title = "PTI (Français)"
source = "https://example.com/feed.xml"
source_type = "rss"
target_lang = "fr"
voice = "fr_montreal"
enabled = true

[[show]]
id = "off"
title = "Disabled"
source = "https://example.com/off.xml"
source_type = "rss"
target_lang = "fr"
enabled = false
"""


def test_load_settings_reads_all_fields(tmp_path: Path):
    p = tmp_path / "settings.toml"
    p.write_text(SETTINGS_TOML, encoding="utf-8")
    s = load_settings(p)
    assert s.transcribe_backend == "mlx-whisper"
    assert s.mlx_whisper_repo == "mlx-community/whisper-large-v3-turbo"
    assert s.translate_backend == "mlx"
    assert s.mlx_llm_repo == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert s.tts_backend == "xtts"
    assert s.tts_device == "cpu"
    assert s.prompts_dir == Path("config/prompts")
    assert s.voices_dir == Path("voices")
    assert s.temperature == 0.3
    assert s.max_tokens == 512
    assert s.gap_ms == 200
    assert s.clip_seconds == 0


def test_load_shows_parses_blocks(tmp_path: Path):
    p = tmp_path / "shows.toml"
    p.write_text(SHOWS_TOML, encoding="utf-8")
    shows = load_shows(p)
    assert len(shows) == 2
    pti = shows[0]
    assert pti.id == "pti-fr"
    assert pti.title == "PTI (Français)"
    assert pti.source == "https://example.com/feed.xml"
    assert pti.source_type == "rss"
    assert pti.target_lang == "fr"
    assert pti.voice == "fr_montreal"
    assert pti.enabled is True
    assert shows[1].voice is None
    assert shows[1].enabled is False


def _settings(tmp_path) -> Settings:
    p = tmp_path / "settings.toml"
    p.write_text(SETTINGS_TOML, encoding="utf-8")
    s = load_settings(p)
    s.prompts_dir = tmp_path / "prompts"
    s.voices_dir = tmp_path / "voices"
    s.prompts_dir.mkdir()
    s.voices_dir.mkdir()
    return s


def _shows(tmp_path):
    p = tmp_path / "shows.toml"
    p.write_text(SHOWS_TOML, encoding="utf-8")
    return load_shows(p)


def test_build_job_resolves_single_wav_voice(tmp_path):
    s = _settings(tmp_path)
    (s.prompts_dir / "fr.txt").write_text("prompt", encoding="utf-8")
    (s.voices_dir / "fr_montreal.wav").write_bytes(b"RIFF")
    job = build_job("pti-fr", settings=s, shows=_shows(tmp_path))
    assert job.show_id == "pti-fr"
    assert job.target_lang == "fr"
    assert job.prompt_path == s.prompts_dir / "fr.txt"
    assert job.voice_refs == [s.voices_dir / "fr_montreal.wav"]


def test_build_job_resolves_folder_of_wavs(tmp_path):
    s = _settings(tmp_path)
    (s.prompts_dir / "fr.txt").write_text("prompt", encoding="utf-8")
    d = s.voices_dir / "fr_montreal"
    d.mkdir()
    (d / "b.wav").write_bytes(b"RIFF")
    (d / "a.wav").write_bytes(b"RIFF")
    job = build_job("pti-fr", settings=s, shows=_shows(tmp_path))
    assert job.voice_refs == [d / "a.wav", d / "b.wav"]


def test_build_job_missing_prompt_raises(tmp_path):
    s = _settings(tmp_path)
    (s.voices_dir / "fr_montreal.wav").write_bytes(b"RIFF")
    with pytest.raises(FileNotFoundError, match="prompt"):
        build_job("pti-fr", settings=s, shows=_shows(tmp_path))


def test_build_job_missing_voice_raises(tmp_path):
    s = _settings(tmp_path)
    (s.prompts_dir / "fr.txt").write_text("prompt", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="voice"):
        build_job("pti-fr", settings=s, shows=_shows(tmp_path))


def test_build_job_unknown_show_raises(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        build_job("nope", settings=s, shows=_shows(tmp_path))
