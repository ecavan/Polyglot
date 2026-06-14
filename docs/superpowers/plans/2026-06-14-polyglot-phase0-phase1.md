# Polyglot Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, manually-runnable single-episode pipeline that turns one English podcast episode into Québécois-French audio (`.mp3`) plus a synced bilingual subtitle file (`.srt`/`.vtt`), with timing derived from the generated audio.

**Architecture:** A generic, config-driven pipeline. `config.py` resolves a `JobSpec` per show. `pipeline.process_episode` runs the stages in order — download → transcribe → translate → synthesize → assemble → subtitles — each stage reading/enriching a shared list of `Segment` dicts. Heavy models (MLX Whisper, MLX LLM, XTTS) are loaded one at a time to bound peak RAM. Deterministic logic (config, subtitle formatting, timeline math, state, feed parsing) is built test-first; thin model wrappers are unit-tested with fakes and exercised for real in the Phase 1 acceptance.

**Tech Stack:** Python 3.12 (via `uv`), `mlx`/`mlx-lm`/`mlx-whisper` (Apple Silicon GPU), `coqui-tts` (XTTS v2, CPU), `feedparser`, `soundfile`, `numpy`, `requests`, `ffmpeg` (system), `pytest` (dev). Spec: `docs/superpowers/specs/2026-06-14-polyglot-media-system-design.md`.

---

## File Structure

Created in this plan:

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project, Python 3.12 pin, deps, `polyglot` console script |
| `src/polyglot/__init__.py` | package marker + version |
| `src/polyglot/config.py` | load/validate TOML → `Settings`, `ShowConfig`, `JobSpec` |
| `src/polyglot/segments.py` | the `Segment` dict shape + `new_segment()` helper |
| `src/polyglot/feeds.py` | `Episode` + `list_episodes()` (RSS via feedparser) |
| `src/polyglot/download.py` | `fetch_audio()` → 16 kHz mono wav (ffmpeg) |
| `src/polyglot/transcribe.py` | `transcribe()` backend dispatch (mlx-whisper default) |
| `src/polyglot/translate.py` | `translate()` backend dispatch (mlx default) |
| `src/polyglot/tts.py` | `synthesize()` backend dispatch (XTTS default, CPU) |
| `src/polyglot/assemble.py` | `build_timeline()`, `assemble()` → mp3 + `EpisodeAudio` |
| `src/polyglot/subtitles.py` | `write_subs()` + SRT/VTT formatting (bilingual + target-only) |
| `src/polyglot/state.py` | `is_done()` / `mark_done()` ledger |
| `src/polyglot/pipeline.py` | `process_episode()` orchestration |
| `src/polyglot/cli.py` | `main()` argparse entrypoint (`show`, `run`) |
| `config/settings.toml` | global settings |
| `config/shows.toml` | per-show config |
| `config/prompts/fr.txt` | French (Montréal) translation system prompt |
| `tests/…` | pytest unit tests mirroring the modules |
| `tests/fixtures/feed.xml` | sample RSS for feed tests |

> Note on layout: the spec diagram shows `cli.py` at repo root; this plan puts it at `src/polyglot/cli.py` so the console entry point `polyglot = "polyglot.cli:main"` resolves cleanly inside the package. This is the one deliberate deviation.

---

## Conventions for every code task

- **TDD loop:** write the failing test → run it red → minimal implementation → run it green → commit.
- **Run tests with:** `uv run pytest <path> -v`
- **Commit format:** `feat:` / `test:` / `chore:` prefixes; small commits.
- A `Segment` is a plain `dict` with exactly these keys (see `segments.py`): `index, start, end, text, translation, speaker, audio_path, audio_dur`.

---

# PHASE 0 — Environment, scaffold, config

**Phase acceptance:** `uv run polyglot show pti-fr` prints the resolved JobSpec (show_id, title, source, target_lang, prompt_path, voice_refs, resolved backends) and exits non-zero with a clear message if the prompt file or named voice is missing.

---

### Task 0.1: Reclaim disk by deleting the broken Qwen download

**Files:** none (one-time ops step).

- [ ] **Step 1: Confirm the blobs are incomplete (safe to delete)**

Run:
```bash
ls -laS ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/ | grep incomplete
```
Expected: four `*.incomplete` files (~3.86G, 3.86G, 3.56G, 1.07G). These are an interrupted download and cannot load.

- [ ] **Step 2: Delete the broken model cache and confirm reclaimed space**

Run:
```bash
df -h /Users/elijahcavan | tail -1
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
df -h /Users/elijahcavan | tail -1
```
Expected: `Avail` increases by ~12 GB.

> No commit — this only touches the HF cache, not the repo.

---

### Task 0.2: Initialize the uv project and package skeleton

**Files:**
- Create: `pyproject.toml`, `src/polyglot/__init__.py`, `src/polyglot/cli.py` (stub), `tests/__init__.py`

- [ ] **Step 1: Init uv project pinned to Python 3.12**

Run:
```bash
cd /Users/elijahcavan/Documents/GitHub/Polyglot
uv init --bare --python 3.12
```
Expected: creates `.python-version` (3.12) and a minimal `pyproject.toml`.

- [ ] **Step 2: Write `pyproject.toml`**

Replace the generated file with:
```toml
[project]
name = "polyglot"
version = "0.1.0"
description = "Local-first podcast/video dubbing pipeline (EN -> target language)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "mlx",
    "mlx-lm",
    "mlx-whisper",
    "coqui-tts[codec]",
    "torchaudio",
    "transformers>=4.57,<5",
    "feedparser",
    "requests",
    "soundfile",
    "numpy",
    "yt-dlp",
    "feedgen",
    "boto3",
]

[project.scripts]
polyglot = "polyglot.cli:main"

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/polyglot"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

> `requires-python` is capped `<3.13` because mlx-whisper breaks on 3.13.
>
> **XTTS dependency notes (learned during execution, torch 2.12 on Apple Silicon):**
> - `coqui-tts[codec]` — torch ≥2.9 needs `torchcodec` for audio I/O (the `codec` extra).
> - `torchaudio` — imported by XTTS's model code; not pulled transitively.
> - `transformers>=4.57,<5` — coqui-tts 0.27.5 imports `isin_mps_friendly` from
>   `transformers.pytorch_utils`, removed in transformers 5.x. Pinning <5 also pins
>   `mlx-lm` to 0.29.x (same stable `load`/`generate`/`make_sampler` API — fine).

- [ ] **Step 3: Create the package + cli stub**

`src/polyglot/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/polyglot/cli.py`:
```python
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="polyglot")
    parser.add_argument("command", nargs="?", help="show | run")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    print(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

`tests/__init__.py`: (empty file)

- [ ] **Step 4: Sync the environment and verify the entry point runs**

Run:
```bash
uv sync
uv run polyglot
```
Expected: argparse help text prints; exit 0. (First `uv sync` will download torch + mlx + coqui-tts; this is the bulk of the ~3–4 GB venv.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version src/polyglot/__init__.py src/polyglot/cli.py tests/__init__.py
git commit -m "chore: scaffold uv project (py3.12) with polyglot console script"
```

---

### Task 0.3: `segments.py` — the Segment shape

**Files:**
- Create: `src/polyglot/segments.py`, `tests/test_segments.py`

- [ ] **Step 1: Write the failing test**

`tests/test_segments.py`:
```python
from polyglot.segments import new_segment, SEGMENT_KEYS


def test_new_segment_has_all_keys_and_defaults():
    seg = new_segment(index=0, start=1.5, end=2.0, text="Hello")
    assert set(seg.keys()) == set(SEGMENT_KEYS)
    assert seg["index"] == 0
    assert seg["start"] == 1.5
    assert seg["end"] == 2.0
    assert seg["text"] == "Hello"
    assert seg["translation"] is None
    assert seg["speaker"] is None
    assert seg["audio_path"] is None
    assert seg["audio_dur"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.segments'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/segments.py`:
```python
SEGMENT_KEYS = (
    "index", "start", "end", "text",
    "translation", "speaker", "audio_path", "audio_dur",
)


def new_segment(index: int, start: float, end: float, text: str) -> dict:
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "translation": None,
        "speaker": None,
        "audio_path": None,
        "audio_dur": None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_segments.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/segments.py tests/test_segments.py
git commit -m "feat: add Segment shape + new_segment helper"
```

---

### Task 0.4: `config.py` — dataclasses + `load_settings`

**Files:**
- Create: `src/polyglot/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from polyglot.config import load_settings

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.config'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/config.py`:
```python
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


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    m, p, h, df = d["models"], d["paths"], d["hosting"], d["defaults"]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/config.py tests/test_config.py
git commit -m "feat: load_settings + Settings dataclass"
```

---

### Task 0.5: `config.py` — `load_shows`

**Files:**
- Modify: `src/polyglot/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_config.py`)**

```python
from polyglot.config import load_shows

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


def test_load_shows_parses_blocks(tmp_path):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_load_shows_parses_blocks -v`
Expected: FAIL — `ImportError: cannot import name 'load_shows'`.

- [ ] **Step 3: Write minimal implementation (append to `src/polyglot/config.py`)**

```python
@dataclass
class ShowConfig:
    id: str
    title: str
    source: str
    source_type: str
    target_lang: str
    voice: str | None
    enabled: bool


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (both config tests).

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/config.py tests/test_config.py
git commit -m "feat: load_shows + ShowConfig dataclass"
```

---

### Task 0.6: `config.py` — `build_job` with voice + prompt resolution

**Files:**
- Modify: `src/polyglot/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_config.py`)**

```python
import pytest
from polyglot.config import build_job, Settings


def _settings(tmp_path) -> Settings:
    p = tmp_path / "settings.toml"
    p.write_text(SETTINGS_TOML, encoding="utf-8")
    s = load_settings(p)
    # repoint dirs into tmp_path so we can create fixtures
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
    assert job.voice_refs == [d / "a.wav", d / "b.wav"]  # sorted


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k build_job -v`
Expected: FAIL — `ImportError: cannot import name 'build_job'`.

- [ ] **Step 3: Write minimal implementation (append to `src/polyglot/config.py`)**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all config tests).

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/config.py tests/test_config.py
git commit -m "feat: build_job with voice + prompt resolution"
```

---

### Task 0.7: Real config files

**Files:**
- Create: `config/settings.toml`, `config/shows.toml`, `config/prompts/fr.txt`

- [ ] **Step 1: Write `config/settings.toml`**

(Use the exact contents from spec §6 `settings.toml`.)

- [ ] **Step 2: Write `config/shows.toml`**

```toml
[[show]]
id          = "pti-fr"
title       = "PTI (Français)"
source      = "https://feeds.megaphone.fm/ESP7239282233"   # PTI (ESPN) — verified 2026-06-14
source_type = "rss"
target_lang = "fr"
# voice omitted for Phase 1 → built-in XTTS speaker. Add `voice = "fr_montreal"` once you
# have a reference clip (see docs/setup/setup-notes.md).
enabled     = true
```
> Feed URL verified on 2026-06-14: returns valid RSS with direct `audio/mpeg` enclosures, latest
> item present. Backups if PTI ever changes: ESPN Daily `https://feeds.megaphone.fm/ESP8348692127`,
> First Take `https://feeds.megaphone.fm/ESP1539938155`. See `docs/setup/setup-notes.md`.
> `voice` is intentionally omitted here so Phase 1 validates the pipeline with a built-in speaker
> before you invest in a cloned voice.

- [ ] **Step 3: Write `config/prompts/fr.txt`**

(Use the exact contents from spec §6 `prompts/fr.txt`.)

- [ ] **Step 4: Commit**

```bash
git add config/
git commit -m "feat: add settings.toml, shows.toml, prompts/fr.txt"
```

---

### Task 0.8: `cli show` command

**Files:**
- Modify: `src/polyglot/cli.py`
- Create: `tests/test_cli_show.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli_show.py`:
```python
from polyglot.cli import cmd_show
from polyglot.config import load_settings, load_shows
from tests.test_config import SETTINGS_TOML, SHOWS_TOML


def _prep(tmp_path):
    sp = tmp_path / "settings.toml"; sp.write_text(SETTINGS_TOML, encoding="utf-8")
    shp = tmp_path / "shows.toml"; shp.write_text(SHOWS_TOML, encoding="utf-8")
    s = load_settings(sp)
    s.prompts_dir = tmp_path / "prompts"; s.voices_dir = tmp_path / "voices"
    s.prompts_dir.mkdir(); s.voices_dir.mkdir()
    (s.prompts_dir / "fr.txt").write_text("p", encoding="utf-8")
    (s.voices_dir / "fr_montreal.wav").write_bytes(b"RIFF")
    return s, load_shows(shp)


def test_cmd_show_prints_jobspec(tmp_path, capsys):
    s, shows = _prep(tmp_path)
    rc = cmd_show("pti-fr", settings=s, shows=shows)
    out = capsys.readouterr().out
    assert rc == 0
    assert "pti-fr" in out
    assert "fr" in out
    assert "fr.txt" in out
    assert "fr_montreal.wav" in out


def test_cmd_show_missing_voice_reports_error(tmp_path, capsys):
    s, shows = _prep(tmp_path)
    (s.voices_dir / "fr_montreal.wav").unlink()
    rc = cmd_show("pti-fr", settings=s, shows=shows)
    out = capsys.readouterr().out + capsys.readouterr().err
    assert rc != 0
    assert "voice" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_show.py -v`
Expected: FAIL — `ImportError: cannot import name 'cmd_show'`.

- [ ] **Step 3: Write implementation** (`src/polyglot/cli.py`)

```python
import argparse
import sys

from polyglot.config import build_job, load_settings, load_shows


def cmd_show(show_id: str, settings=None, shows=None) -> int:
    try:
        job = build_job(show_id, settings=settings, shows=shows)
    except (FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    s = job.settings
    print(f"show_id        : {job.show_id}")
    print(f"title          : {job.title}")
    print(f"source         : {job.source}  ({job.source_type})")
    print(f"target_lang    : {job.target_lang}")
    print(f"prompt_path    : {job.prompt_path}")
    print(f"voice_refs     : {[str(p) for p in job.voice_refs] or '(built-in voice)'}")
    print(f"transcribe     : {s.transcribe_backend} ({s.mlx_whisper_repo})")
    print(f"translate      : {s.translate_backend} ({s.mlx_llm_repo})")
    print(f"tts            : {s.tts_backend} (device={s.tts_device})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="polyglot")
    sub = parser.add_subparsers(dest="command")
    p_show = sub.add_parser("show", help="print resolved JobSpec for a show")
    p_show.add_argument("show_id")
    args = parser.parse_args()

    if args.command == "show":
        return cmd_show(args.show_id)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_show.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/cli.py tests/test_cli_show.py
git commit -m "feat: polyglot show command"
```

---

### Task 0.9: Pre-download models + PHASE 0 ACCEPTANCE

**Files:** none (ops + acceptance).

- [ ] **Step 1: Pre-download the MLX models so Phase 1 isn't gated on network**

Run:
```bash
uv run python -c "from huggingface_hub import snapshot_download as d; \
d('mlx-community/Qwen2.5-7B-Instruct-4bit'); \
d('mlx-community/whisper-large-v3-turbo'); print('models cached')"
```
Expected: `models cached`; ~5.9 GB downloaded to `~/.cache/huggingface/hub/`.

- [ ] **Step 2: Run the Phase 0 acceptance check**

Run:
```bash
uv run polyglot show pti-fr
```
Expected: prints the resolved JobSpec with `prompt_path` ending `config/prompts/fr.txt`, `voice_refs` showing the resolved wav(s) (or `(built-in voice)` if you haven't added one yet), and the three backends. Exits 0.

- [ ] **Step 3: Verify the error path**

Run:
```bash
mv config/prompts/fr.txt config/prompts/fr.txt.bak && uv run polyglot show pti-fr; \
mv config/prompts/fr.txt.bak config/prompts/fr.txt
```
Expected: non-zero exit, message naming the missing prompt path.

> Phase 0 complete. Do not start Phase 1 until `polyglot show pti-fr` passes.

---

# PHASE 1 — Core single-episode pipeline (the quality gate)

**Phase acceptance:** `uv run polyglot run pti-fr --latest --clip-seconds 180` produces `output/audio/pti-fr/<ep>.mp3` and a synced bilingual `output/subs/pti-fr/<ep>.srt`. **You listen and judge:** is the French good enough to enjoy daily, and do the subtitles line up?

---

### Task 1.1: `feeds.py` — list episodes from RSS

**Files:**
- Create: `src/polyglot/feeds.py`, `tests/test_feeds.py`, `tests/fixtures/feed.xml`

- [ ] **Step 1: Create the fixture feed**

`tests/fixtures/feed.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test Show</title>
  <item>
    <title>Episode Two</title>
    <guid isPermaLink="false">guid-2</guid>
    <pubDate>Tue, 10 Jun 2026 10:00:00 +0000</pubDate>
    <enclosure url="https://example.com/ep2.mp3" length="1000" type="audio/mpeg"/>
  </item>
  <item>
    <title>Episode One</title>
    <guid isPermaLink="false">guid-1</guid>
    <pubDate>Mon, 09 Jun 2026 10:00:00 +0000</pubDate>
    <enclosure url="https://example.com/ep1.mp3" length="2000" type="audio/mpeg"/>
  </item>
</channel></rss>
```

- [ ] **Step 2: Write the failing test**

`tests/test_feeds.py`:
```python
from pathlib import Path
from polyglot.feeds import list_episodes_from_url

FIXTURE = Path(__file__).parent / "fixtures" / "feed.xml"


def test_list_episodes_parses_rss():
    eps = list_episodes_from_url(FIXTURE.as_uri(), limit=None)
    assert len(eps) == 2
    assert eps[0].guid == "guid-2"
    assert eps[0].title == "Episode Two"
    assert eps[0].media_url == "https://example.com/ep2.mp3"
    assert eps[0].published is not None


def test_list_episodes_respects_limit():
    eps = list_episodes_from_url(FIXTURE.as_uri(), limit=1)
    assert len(eps) == 1
    assert eps[0].guid == "guid-2"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_feeds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.feeds'`.

- [ ] **Step 4: Write minimal implementation**

`src/polyglot/feeds.py`:
```python
from dataclasses import dataclass

import feedparser

from polyglot.config import JobSpec


@dataclass
class Episode:
    guid: str
    title: str
    published: str | None
    media_url: str


def _episode_from_entry(e) -> Episode | None:
    enclosures = e.get("enclosures") or []
    if not enclosures:
        return None
    media_url = enclosures[0].get("href")
    if not media_url:
        return None
    guid = e.get("id") or e.get("guid") or media_url
    return Episode(
        guid=guid,
        title=e.get("title", "(untitled)"),
        published=e.get("published"),
        media_url=media_url,
    )


def list_episodes_from_url(url: str, limit: int | None) -> list[Episode]:
    parsed = feedparser.parse(url)
    out: list[Episode] = []
    for e in parsed.entries:
        ep = _episode_from_entry(e)
        if ep is not None:
            out.append(ep)
        if limit is not None and len(out) >= limit:
            break
    return out


def list_episodes(job: JobSpec, limit: int | None) -> list[Episode]:
    if job.source_type == "rss":
        return list_episodes_from_url(job.source, limit)
    raise NotImplementedError(f"source_type '{job.source_type}' not supported in Phase 1")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_feeds.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/polyglot/feeds.py tests/test_feeds.py tests/fixtures/feed.xml
git commit -m "feat: feeds.list_episodes (RSS)"
```

---

### Task 1.2: `download.py` — fetch + normalize to 16 kHz mono wav

**Files:**
- Create: `src/polyglot/download.py`, `tests/test_download.py`

- [ ] **Step 1: Write the failing test (builds the ffmpeg arg list; pure function)**

`tests/test_download.py`:
```python
from pathlib import Path
from polyglot.download import ffmpeg_normalize_cmd


def test_ffmpeg_cmd_full_episode():
    cmd = ffmpeg_normalize_cmd(Path("in.mp3"), Path("out.wav"), clip_seconds=0)
    assert cmd[0] == "ffmpeg"
    assert "-t" not in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "16000" in cmd
    assert cmd[-1] == "out.wav"


def test_ffmpeg_cmd_trimmed():
    cmd = ffmpeg_normalize_cmd(Path("in.mp3"), Path("out.wav"), clip_seconds=180)
    i = cmd.index("-t")
    assert cmd[i + 1] == "180"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.download'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/download.py`:
```python
import subprocess
from pathlib import Path

import requests


def ffmpeg_normalize_cmd(src: Path, dst: Path, clip_seconds: int) -> list[str]:
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", str(clip_seconds)]
    cmd += ["-ac", "1", "-ar", "16000", str(dst)]
    return cmd


def _download_to(url: str, dst: Path) -> Path:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dst


def fetch_audio(media_url: str, out_dir: Path, clip_seconds: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "source_raw"
    _download_to(media_url, raw)
    wav = out_dir / "source_16k_mono.wav"
    subprocess.run(ffmpeg_normalize_cmd(raw, wav, clip_seconds), check=True)
    return wav
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/download.py tests/test_download.py
git commit -m "feat: download.fetch_audio + ffmpeg normalize"
```

---

### Task 1.3: `transcribe.py` — backend dispatch + segment shaping

**Files:**
- Create: `src/polyglot/transcribe.py`, `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing test (shape the mlx-whisper result; pure function)**

`tests/test_transcribe.py`:
```python
from polyglot.transcribe import segments_from_mlx_result
from polyglot.segments import SEGMENT_KEYS


def test_segments_from_mlx_result():
    result = {
        "text": "Hello there. How are you?",
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": " Hello there."},
            {"id": 1, "start": 1.2, "end": 2.8, "text": " How are you?"},
        ],
    }
    segs = segments_from_mlx_result(result)
    assert len(segs) == 2
    assert set(segs[0].keys()) == set(SEGMENT_KEYS)
    assert segs[0]["index"] == 0
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.2
    assert segs[0]["text"] == "Hello there."   # stripped
    assert segs[1]["index"] == 1
    assert segs[1]["text"] == "How are you?"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.transcribe'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/transcribe.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/transcribe.py tests/test_transcribe.py
git commit -m "feat: transcribe backend dispatch + mlx segment shaping"
```

---

### Task 1.4: `translate.py` — prompt building + backend dispatch

**Files:**
- Create: `src/polyglot/translate.py`, `tests/test_translate.py`

- [ ] **Step 1: Write the failing test (message building + loop with injected generator)**

`tests/test_translate.py`:
```python
from polyglot.translate import build_messages, translate_with
from polyglot.segments import new_segment


def test_build_messages_includes_context():
    msgs = build_messages("SYSTEM", "current line", prev_text="previous line")
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert "previous line" in msgs[1]["content"]
    assert "current line" in msgs[1]["content"]


def test_build_messages_no_context():
    msgs = build_messages("SYSTEM", "only line", prev_text=None)
    assert msgs[1]["content"] == "only line"


def test_translate_with_fake_generator():
    segs = [
        new_segment(0, 0.0, 1.0, "Hello"),
        new_segment(1, 1.0, 2.0, "World"),
    ]

    def fake_generate(messages):
        # echo a fake french translation of the last user line
        user = messages[-1]["content"]
        return f"FR[{user.splitlines()[-1]}]"

    out = translate_with(segs, system="SYS", generate=fake_generate)
    assert out[0]["translation"] == "FR[Hello]"
    assert out[1]["translation"] == "FR[World]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_translate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.translate'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/translate.py`:
```python
import gc
from typing import Callable

from polyglot.config import JobSpec, Settings


def build_messages(system: str, text: str, prev_text: str | None) -> list[dict]:
    if prev_text:
        user = f"[contexte précédent: {prev_text}]\n{text}"
    else:
        user = text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def translate_with(
    segments: list[dict],
    system: str,
    generate: Callable[[list[dict]], str],
) -> list[dict]:
    prev = None
    for seg in segments:
        msgs = build_messages(system, seg["text"], prev_text=prev)
        seg["translation"] = generate(msgs).strip()
        prev = seg["text"]
    return segments


def _mlx_generator(settings: Settings):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(settings.mlx_llm_repo)
    sampler = make_sampler(temp=settings.temperature)

    def gen(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            model, tokenizer, prompt=prompt,
            max_tokens=settings.max_tokens, sampler=sampler, verbose=False,
        )

    def release():
        nonlocal model, tokenizer
        del model
        del tokenizer
        gc.collect()

    return gen, release


def _ollama_generator(settings: Settings):
    import requests

    def gen(messages: list[dict]) -> str:
        r = requests.post(settings.ollama_url, json={
            "model": settings.ollama_model, "stream": False,
            "options": {"temperature": settings.temperature},
            "messages": messages,
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]

    return gen, (lambda: None)


def translate(segments: list[dict], job: JobSpec, settings: Settings) -> list[dict]:
    system = job.prompt_path.read_text(encoding="utf-8")
    if settings.translate_backend == "mlx":
        gen, release = _mlx_generator(settings)
    elif settings.translate_backend == "ollama":
        gen, release = _ollama_generator(settings)
    else:
        raise ValueError(f"unknown translate_backend: {settings.translate_backend}")
    try:
        return translate_with(segments, system, gen)
    finally:
        release()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_translate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/translate.py tests/test_translate.py
git commit -m "feat: translate with context + mlx/ollama backends"
```

---

### Task 1.5: `tts.py` — synthesis + duration capture

**Files:**
- Create: `src/polyglot/tts.py`, `tests/test_tts.py`

- [ ] **Step 1: Write the failing test (loop fills audio_path + audio_dur via injected synth)**

`tests/test_tts.py`:
```python
from pathlib import Path
import numpy as np
import soundfile as sf
from polyglot.tts import synthesize_with, SR
from polyglot.segments import new_segment


def test_synthesize_with_fake_writes_clips_and_durations(tmp_path: Path):
    segs = [new_segment(0, 0.0, 1.0, "Bonjour")]
    segs[0]["translation"] = "Bonjour"

    def fake_synth(text):
        # 0.5 s of silence at SR
        return np.zeros(int(0.5 * SR), dtype=np.float32)

    out = synthesize_with(segs, fake_synth, out_dir=tmp_path)
    p = Path(out[0]["audio_path"])
    assert p.exists()
    assert abs(out[0]["audio_dur"] - 0.5) < 1e-6
    data, sr = sf.read(p)
    assert sr == SR
    assert len(data) == int(0.5 * SR)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.tts'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/tts.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/tts.py tests/test_tts.py
git commit -m "feat: tts.synthesize (XTTS, cpu) + duration capture"
```

---

### Task 1.6: `assemble.py` — timeline math + concat + mp3

**Files:**
- Create: `src/polyglot/assemble.py`, `tests/test_assemble.py`

- [ ] **Step 1: Write the failing tests (timeline + concat are pure/deterministic)**

`tests/test_assemble.py`:
```python
from pathlib import Path
import numpy as np
import soundfile as sf
from polyglot.assemble import build_timeline, concat_audio
from polyglot.tts import SR
from polyglot.segments import new_segment


def _seg(i, dur):
    s = new_segment(i, 0.0, 0.0, f"t{i}")
    s["audio_dur"] = dur
    return s


def test_build_timeline_accumulates_with_gap():
    segs = [_seg(0, 1.0), _seg(1, 2.0)]
    tl = build_timeline(segs, gap_ms=200)
    assert tl[0] == (0.0, 1.0)
    # second cue starts after first dur + 0.2 gap
    assert abs(tl[1][0] - 1.2) < 1e-9
    assert abs(tl[1][1] - 3.2) < 1e-9


def test_concat_audio_inserts_silence(tmp_path: Path):
    a = tmp_path / "a.wav"; b = tmp_path / "b.wav"
    sf.write(a, np.ones(SR, dtype=np.float32), SR, subtype="FLOAT")        # 1.0 s
    sf.write(b, np.ones(SR * 2, dtype=np.float32), SR, subtype="FLOAT")    # 2.0 s
    segs = [new_segment(0, 0, 0, "a"), new_segment(1, 0, 0, "b")]
    segs[0]["audio_path"] = str(a); segs[1]["audio_path"] = str(b)
    full = concat_audio(segs, gap_ms=200)
    gap_samples = int(0.2 * SR)
    assert len(full) == SR + gap_samples + SR * 2 + gap_samples
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_assemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.assemble'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/assemble.py`:
```python
import subprocess
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
    gap = gap_ms / 1000.0
    timeline: list[tuple[float, float]] = []
    t = 0.0
    for seg in segments:
        dur = seg["audio_dur"]
        timeline.append((t, t + dur))
        t += dur + gap
    return timeline


def concat_audio(segments: list[dict], gap_ms: int) -> np.ndarray:
    gap = np.zeros(int(gap_ms / 1000.0 * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    for seg in segments:
        data, _sr = sf.read(seg["audio_path"], dtype="float32")
        parts.append(data)
        parts.append(gap)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def assemble(segments: list[dict], out_mp3: Path, settings: Settings) -> EpisodeAudio:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    full = concat_audio(segments, settings.gap_ms)
    tmp_wav = out_mp3.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), full, SR, subtype="FLOAT")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
        check=True,
    )
    tmp_wav.unlink(missing_ok=True)
    return EpisodeAudio(
        duration=len(full) / SR,
        byte_length=out_mp3.stat().st_size,
        timeline=build_timeline(segments, settings.gap_ms),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_assemble.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/assemble.py tests/test_assemble.py
git commit -m "feat: assemble timeline + concat + mp3 encode"
```

---

### Task 1.7: `subtitles.py` — SRT/VTT, bilingual + target-only

**Files:**
- Create: `src/polyglot/subtitles.py`, `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_subtitles.py`:
```python
from pathlib import Path
from polyglot.subtitles import srt_timestamp, vtt_timestamp, build_srt, write_subs
from polyglot.segments import new_segment


def test_srt_timestamp():
    assert srt_timestamp(0.0) == "00:00:00,000"
    assert srt_timestamp(3661.5) == "01:01:01,500"


def test_vtt_timestamp():
    assert vtt_timestamp(3661.5) == "01:01:01.500"


def _segs():
    a = new_segment(0, 0, 0, "Hello"); a["translation"] = "Bonjour"
    b = new_segment(1, 0, 0, "Bye"); b["translation"] = "Salut"
    return [a, b]


def test_build_srt_bilingual():
    tl = [(0.0, 1.0), (1.2, 2.0)]
    out = build_srt(_segs(), tl, bilingual=True)
    assert "1\n00:00:00,000 --> 00:00:01,000" in out
    assert "Bonjour" in out and "Hello" in out
    assert "2\n00:00:01,200 --> 00:00:02,000" in out


def test_build_srt_target_only_excludes_source():
    tl = [(0.0, 1.0), (1.2, 2.0)]
    out = build_srt(_segs(), tl, bilingual=False)
    assert "Bonjour" in out
    assert "Hello" not in out


def test_write_subs_creates_four_files(tmp_path: Path):
    tl = [(0.0, 1.0), (1.2, 2.0)]
    write_subs(_segs(), tl, out_dir=tmp_path, show_id="s", ep_id="e")
    assert (tmp_path / "e.srt").exists()
    assert (tmp_path / "e.vtt").exists()
    assert (tmp_path / "e.target.srt").exists()
    assert (tmp_path / "e.target.vtt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subtitles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.subtitles'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/subtitles.py`:
```python
from pathlib import Path


def _hms(seconds: float) -> tuple[int, int, int, int]:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return h, m, s, ms


def srt_timestamp(seconds: float) -> str:
    h, m, s, ms = _hms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    h, m, s, ms = _hms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _lines_for(seg: dict, bilingual: bool) -> list[str]:
    prefix = f"{seg['speaker']}: " if seg.get("speaker") else ""
    lines = [f"{prefix}{seg['translation']}"]
    if bilingual:
        lines.append(seg["text"])
    return lines


def build_srt(segments: list[dict], timeline: list[tuple[float, float]], bilingual: bool) -> str:
    blocks = []
    for n, (seg, (start, end)) in enumerate(zip(segments, timeline), start=1):
        body = "\n".join(_lines_for(seg, bilingual))
        blocks.append(f"{n}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{body}\n")
    return "\n".join(blocks)


def build_vtt(segments: list[dict], timeline: list[tuple[float, float]], bilingual: bool) -> str:
    blocks = ["WEBVTT\n"]
    for seg, (start, end) in zip(segments, timeline):
        body = "\n".join(_lines_for(seg, bilingual))
        blocks.append(f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}\n{body}\n")
    return "\n".join(blocks)


def write_subs(segments, timeline, out_dir: Path, show_id: str, ep_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ep_id}.srt").write_text(build_srt(segments, timeline, True), encoding="utf-8")
    (out_dir / f"{ep_id}.vtt").write_text(build_vtt(segments, timeline, True), encoding="utf-8")
    (out_dir / f"{ep_id}.target.srt").write_text(build_srt(segments, timeline, False), encoding="utf-8")
    (out_dir / f"{ep_id}.target.vtt").write_text(build_vtt(segments, timeline, False), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_subtitles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/subtitles.py tests/test_subtitles.py
git commit -m "feat: subtitles SRT/VTT (bilingual + target-only) on dubbed timeline"
```

---

### Task 1.8: `pipeline.py` — orchestrate one episode

**Files:**
- Create: `src/polyglot/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test (orchestration with monkeypatched stages)**

`tests/test_pipeline.py`:
```python
from pathlib import Path
import numpy as np
import polyglot.pipeline as pipeline
from polyglot.config import JobSpec, Settings
from polyglot.feeds import Episode
from polyglot.segments import new_segment


def _settings(tmp_path) -> Settings:
    return Settings(
        transcribe_backend="mlx-whisper", mlx_whisper_repo="r", faster_whisper="m",
        translate_backend="mlx", mlx_llm_repo="r", ollama_model="m", ollama_url="u",
        tts_backend="xtts", tts_device="cpu",
        cache_dir=tmp_path / "cache", output_dir=tmp_path / "output",
        voices_dir=tmp_path / "voices", prompts_dir=tmp_path / "prompts",
        state_path=tmp_path / "state.json",
        hosting_type="r2", public_base_url="x", bucket="b",
        clip_seconds=0, diarize=False, temperature=0.3, max_tokens=512, gap_ms=200,
    )


def test_process_episode_writes_outputs(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    (tmp_path / "prompts").mkdir(); (tmp_path / "prompts" / "fr.txt").write_text("p")
    job = JobSpec("pti-fr", "PTI", "src", "rss", "fr",
                  tmp_path / "prompts" / "fr.txt", [], s)
    ep = Episode(guid="g1", title="Ep 1", published=None, media_url="http://x/ep.mp3")

    monkeypatch.setattr(pipeline.download, "fetch_audio", lambda *a, **k: tmp_path / "in.wav")

    def fake_transcribe(wav, settings):
        return [new_segment(0, 0, 1, "Hello"), new_segment(1, 1, 2, "Bye")]
    monkeypatch.setattr(pipeline.transcribe, "transcribe", fake_transcribe)

    def fake_translate(segs, job, settings):
        for sg in segs:
            sg["translation"] = "FR-" + sg["text"]
        return segs
    monkeypatch.setattr(pipeline.translate, "translate", fake_translate)

    def fake_synth(segs, job, settings, out_dir):
        import soundfile as sf
        out_dir.mkdir(parents=True, exist_ok=True)
        for sg in segs:
            p = out_dir / f"seg_{sg['index']}.wav"
            sf.write(str(p), np.zeros(24000, dtype=np.float32), 24000, subtype="FLOAT")
            sg["audio_path"] = str(p); sg["audio_dur"] = 1.0
        return segs
    monkeypatch.setattr(pipeline.tts, "synthesize", fake_synth)

    out = pipeline.process_episode(job, ep, s)
    assert Path(out["mp3"]).exists()
    assert Path(out["srt"]).exists()
    assert out["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyglot.pipeline'`.

- [ ] **Step 3: Write minimal implementation**

`src/polyglot/pipeline.py`:
```python
import re
import traceback
from pathlib import Path

from polyglot import assemble, download, subtitles, transcribe, translate, tts
from polyglot.config import JobSpec, Settings
from polyglot.feeds import Episode


def _safe_id(guid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", guid)[:120]


def process_episode(job: JobSpec, episode: Episode, settings: Settings) -> dict:
    ep_id = _safe_id(episode.guid)
    work = settings.cache_dir / job.show_id / ep_id
    audio_dir = settings.output_dir / "audio" / job.show_id
    subs_dir = settings.output_dir / "subs" / job.show_id
    out_mp3 = audio_dir / f"{ep_id}.mp3"
    try:
        wav = download.fetch_audio(episode.media_url, work, settings.clip_seconds)
        segments = transcribe.transcribe(wav, settings)
        segments = translate.translate(segments, job, settings)   # loads LLM, frees it
        segments = tts.synthesize(segments, job, settings, work / "segments")
        audio = assemble.assemble(segments, out_mp3, settings)
        subtitles.write_subs(segments, audio.timeline, subs_dir, job.show_id, ep_id)
        return {
            "ok": True,
            "mp3": str(out_mp3),
            "srt": str(subs_dir / f"{ep_id}.srt"),
            "duration": audio.duration,
            "byte_length": audio.byte_length,
        }
    except Exception as e:  # episode isolation: log + skip
        traceback.print_exc()
        return {"ok": False, "error": str(e), "guid": episode.guid}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polyglot/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline.process_episode orchestration with episode isolation"
```

---

### Task 1.9: `cli run` command

**Files:**
- Modify: `src/polyglot/cli.py`
- Create: `tests/test_cli_run.py`

- [ ] **Step 1: Write the failing test (selection logic; pure)**

`tests/test_cli_run.py`:
```python
from polyglot.cli import select_episode
from polyglot.feeds import Episode


def _eps():
    return [
        Episode("g2", "Two", None, "http://x/2.mp3"),
        Episode("g1", "One", None, "http://x/1.mp3"),
    ]


def test_select_latest():
    ep = select_episode(_eps(), latest=True, url=None)
    assert ep.guid == "g2"


def test_select_by_url():
    ep = select_episode(_eps(), latest=False, url="http://manual/x.mp3")
    assert ep.media_url == "http://manual/x.mp3"
    assert ep.guid  # synthesized guid is non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_episode'`.

- [ ] **Step 3: Write implementation (modify `src/polyglot/cli.py`)**

Add imports near the top:
```python
from polyglot import feeds, pipeline
from polyglot.feeds import Episode
```

Add the selection helper and `cmd_run`:
```python
def select_episode(episodes, latest: bool, url: str | None) -> Episode:
    if url:
        import hashlib
        guid = "manual-" + hashlib.sha1(url.encode()).hexdigest()[:12]
        return Episode(guid=guid, title="(manual)", published=None, media_url=url)
    if not episodes:
        raise ValueError("no episodes found in feed")
    return episodes[0]  # feeds are newest-first


def cmd_run(show_id: str, latest: bool, url: str | None, file: str | None,
            clip_seconds: int | None) -> int:
    job = build_job(show_id)
    if clip_seconds is not None:
        job.settings.clip_seconds = clip_seconds
    if file:
        ep = Episode(guid=f"file-{file}", title="(file)", published=None, media_url=file)
    else:
        episodes = feeds.list_episodes(job, limit=5)
        ep = select_episode(episodes, latest=latest, url=url)
    result = pipeline.process_episode(job, ep, job.settings)
    if result["ok"]:
        print(f"OK  mp3: {result['mp3']}")
        print(f"    srt: {result['srt']}")
        print(f"    duration: {result['duration']:.1f}s")
        return 0
    print(f"FAILED: {result['error']}", file=sys.stderr)
    return 1
```

Wire it into `main()` after the `show` subparser:
```python
    p_run = sub.add_parser("run", help="process one episode end-to-end")
    p_run.add_argument("show_id")
    p_run.add_argument("--latest", action="store_true")
    p_run.add_argument("--url")
    p_run.add_argument("--file")
    p_run.add_argument("--clip-seconds", type=int, default=None)
```
and in the dispatch:
```python
    if args.command == "run":
        return cmd_run(args.show_id, args.latest, args.url, args.file, args.clip_seconds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_run.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/polyglot/cli.py tests/test_cli_run.py
git commit -m "feat: polyglot run command + episode selection"
```

---

### Task 1.10: PHASE 1 ACCEPTANCE — real run + listen

**Files:** none (acceptance). Prerequisites: a verified feed URL in `config/shows.toml` and (optionally) a reference clip at `voices/fr_montreal.wav`.

- [ ] **Step 1: Confirm the feed resolves**

Run: `uv run python -c "from polyglot.config import build_job; from polyglot import feeds; j=build_job('pti-fr'); print([e.title for e in feeds.list_episodes(j, 3)])"`
Expected: a list of recent episode titles (proves the feed URL + enclosures work).

- [ ] **Step 2: Run the pipeline on a 3-minute clip**

Run: `uv run polyglot run pti-fr --latest --clip-seconds 180`
Expected: `OK  mp3: output/audio/pti-fr/<ep>.mp3` and a `.srt` path; total runtime dominated by XTTS (CPU). If XTTS is the bottleneck, that's expected.

- [ ] **Step 3: Listen and judge (the quality gate)**

- Play `output/audio/pti-fr/<ep>.mp3`.
- Open the `.srt` in a player alongside, or inspect cue timings.
- **Judge:** Is the Québécois French natural and enjoyable? Do the subtitle cues line up with the spoken French (they should, since timing is derived from generated audio)?
- If French quality is weak: iterate on `config/prompts/fr.txt`, try `temperature` 0.2–0.4, or switch `translate_backend`/model.
- If timing is off: inspect `build_timeline` output and `gap_ms`.

> **Do not proceed to Phase 2+ until this clears your bar.** Everything after is plumbing.

---

## Self-Review (completed by plan author)

- **Spec coverage:** config.py (Tasks 0.4–0.6) ✓; feeds (1.1) ✓; download (1.2) ✓; transcribe mlx+faster (1.3) ✓; translate mlx+ollama (1.4) ✓; tts XTTS w/ both shims + cpu (1.5) ✓; assemble timeline+mp3 (1.6) ✓; subtitles bilingual+target SRT/VTT on dubbed timeline (1.7) ✓; pipeline w/ isolation + load-one-model-at-a-time (1.8) ✓; cli show+run (0.8, 1.9) ✓; disk reclaim (0.1) ✓; Python 3.12/uv (0.2) ✓. **Deferred to later plans (by design):** state.py, publish_rss.py, watch/cron, storage cleanup (Phases 3–4); diarization/video/fine-tune (Phase 5).
- **Placeholder scan:** none. The PTI feed URL is verified and inlined in Task 0.7; the voice is intentionally omitted for Phase 1 (built-in speaker) per the spec's quality-gate-first discipline. Reference-clip guidance lives in `docs/setup/setup-notes.md`.
- **Type consistency:** `Settings`/`ShowConfig`/`JobSpec` fields are referenced consistently across config, cli, pipeline; `SR=24000` defined once in `tts.py` and imported by `assemble.py`/tests; `Segment` keys via `new_segment`/`SEGMENT_KEYS` everywhere; `Episode` fields consistent across feeds/cli/pipeline tests.

---

## Subsequent plans (drafted after the Phase 1 quality gate clears)

1. **Phase 2 — Multi-language + cloning:** add `config/prompts/it.txt` + an `it` show; verify `voice_refs` cloning (single clip + folder). Pure config; minimal code.
2. **Phase 3 — Distribution:** `state.py` ledger, `publish_rss.py` (feedgen + boto3→R2 upload). Requires the R2 account/bucket/domain from setup notes and the three `R2_*` env vars.
3. **Phase 4 — Automation + storage:** new-episode detection vs ledger, `watch --once`, `storage.py` cleanup, cron entry.
4. **Phase 5 — Optional:** diarization, video mux + YouTube, XTTS fine-tuning.
